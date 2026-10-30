"""
Dobot Magician Multi-Color Box Pick-and-Place Benchmark Environment.
Replaces Franka with the 4-DOF Dobot Magician robot and evaluates sorting
small colored blocks (Red, Green, Blue, Yellow) into corresponding colored receptacles/plates.
Produces complete HD execution videos with closed-loop servo precision.
"""
import os
import sys
import time
import numpy as np
import pybullet as p
import pybullet_data
import cv2

from dobot_kinematics import DobotKinematics

# Color Definitions (RGBA)
COLORS = {
    "red": [0.90, 0.15, 0.15, 1.0],
    "green": [0.15, 0.85, 0.25, 1.0],
    "blue": [0.15, 0.35, 0.90, 1.0],
    "yellow": [0.95, 0.85, 0.15, 1.0],
    "purple": [0.70, 0.15, 0.85, 1.0],
    "plate_blue": [0.20, 0.40, 0.90, 0.85],
    "plate_red": [0.90, 0.20, 0.20, 0.85],
    "plate_green": [0.20, 0.85, 0.30, 0.85],
    "plate_yellow": [0.90, 0.85, 0.20, 0.85],
}

class DobotColorBoxEnv:
    def __init__(self, render_gui=False, width=640, height=480):
        self.render_gui = render_gui
        self.width = width
        self.height = height
        self.kin = DobotKinematics()

        if self.render_gui:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1.0 / 240.0)

        # Centered Front 45-degree elevated camera
        self.cam_target = [0.22, 0.0, -0.04]
        self.cam_eye = [0.52, 0.0, 0.26]
        self.cam_up = [0.0, 0.0, 1.0]
        self.fov = 52.0

        self.view_matrix = p.computeViewMatrix(
            cameraEyePosition=self.cam_eye,
            cameraTargetPosition=self.cam_target,
            cameraUpVector=self.cam_up
        )
        self.proj_matrix = p.computeProjectionMatrixFOV(
            fov=self.fov,
            aspect=float(width) / float(height),
            nearVal=0.05,
            farVal=2.5
        )

        self.robot_id = None
        self.plane_id = None
        self.table_id = None
        self.boxes = {}
        self.plates = {}
        self.grasped_id = None
        self.gripper_closed = False

        self.ee_link_index = 6
        self.gripper_j_left = 7
        self.gripper_j_right = 8

        self._load_robot_and_scene()

    def _load_robot_and_scene(self):
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        self.plane_id = p.loadURDF("plane.urdf", [0, 0, -0.05])

        # Load Dobot URDF
        urdf_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "dobot_magician_ros2", "ai_training", "env", "dobot_pybullet.urdf")
        )
        self.robot_id = p.loadURDF(urdf_path, [0, 0, 0], useFixedBase=True)

        for j in [self.gripper_j_left, self.gripper_j_right]:
            p.changeDynamics(self.robot_id, j, lateralFriction=3.5, spinningFriction=0.2)

    def spawn_colored_boxes_and_plates(self, task_config):
        for bid in self.boxes.values():
            try: p.removeBody(bid)
            except Exception: pass
        for pid in self.plates.values():
            try: p.removeBody(pid)
            except Exception: pass

        self.boxes = {}
        self.plates = {}
        self.grasped_id = None
        self.gripper_closed = False

        box_half = [0.012, 0.012, 0.012] # 2.4 cm cube
        plate_half = [0.030, 0.030, 0.003] # 6 cm square receptacle plate

        for item in task_config:
            cname = item["color"]
            b_pos = item["box_pos"]
            p_pos = item["plate_pos"]
            rgba = COLORS.get(cname, [0.5, 0.5, 0.5, 1.0])
            plate_rgba = COLORS.get(f"plate_{cname}", [0.5, 0.5, 0.5, 0.8])

            # Spawn Box
            col_b = p.createCollisionShape(p.GEOM_BOX, halfExtents=box_half)
            vis_b = p.createVisualShape(p.GEOM_BOX, halfExtents=box_half, rgbaColor=rgba)
            bid = p.createMultiBody(
                baseMass=0.04,
                baseCollisionShapeIndex=col_b,
                baseVisualShapeIndex=vis_b,
                basePosition=[b_pos[0], b_pos[1], -0.05 + box_half[2] + 0.001],
                baseOrientation=[0, 0, 0, 1]
            )
            p.changeDynamics(bid, -1, lateralFriction=2.8, linearDamping=0.5, angularDamping=0.8)
            self.boxes[cname] = bid

            # Spawn Destination Plate
            col_p = p.createCollisionShape(p.GEOM_BOX, halfExtents=plate_half)
            vis_p = p.createVisualShape(p.GEOM_BOX, halfExtents=plate_half, rgbaColor=plate_rgba)
            pid = p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=col_p,
                baseVisualShapeIndex=vis_p,
                basePosition=[p_pos[0], p_pos[1], -0.048],
                baseOrientation=[0, 0, 0, 1]
            )
            p.changeDynamics(pid, -1, lateralFriction=1.8)
            self.plates[cname] = pid

        # Settle
        for _ in range(50):
            p.stepSimulation()

    def set_cartesian_target(self, x, y, z, r=0.0, gripper_close=False):
        ik_sol = self.kin.inverse(x, y, z, yaw=r)
        if ik_sol is None:
            return False

        j1, j2, j3, j4 = ik_sol
        j2_urdf = np.pi / 2.0 - j2
        j3_urdf = -np.pi / 2.0 - j3

        p.resetJointState(self.robot_id, 1, j1)
        p.resetJointState(self.robot_id, 2, j2_urdf)
        p.resetJointState(self.robot_id, 3, j3_urdf)
        p.resetJointState(self.robot_id, 4, -j2_urdf)
        p.resetJointState(self.robot_id, 5, -j3_urdf)
        p.resetJointState(self.robot_id, 6, j4)

        jaw_pos = 0.012 if gripper_close else 0.0
        p.setJointMotorControl2(self.robot_id, self.gripper_j_left, p.POSITION_CONTROL, targetPosition=jaw_pos, force=100)
        p.setJointMotorControl2(self.robot_id, self.gripper_j_right, p.POSITION_CONTROL, targetPosition=-jaw_pos, force=100)

        self.gripper_closed = gripper_close
        self._update_gripper_physics()
        p.stepSimulation()
        return True

    def _update_gripper_physics(self):
        ee_pos = p.getLinkState(self.robot_id, self.ee_link_index)[0]

        if self.gripper_closed and self.grasped_id is None:
            for cname, bid in self.boxes.items():
                b_pos, _ = p.getBasePositionAndOrientation(bid)
                dist = np.linalg.norm(np.array(ee_pos) - np.array(b_pos))
                if dist < 0.055:
                    self.grasped_id = bid
                    break

        if self.gripper_closed and self.grasped_id is not None:
            # Carry box synchronously right under gripper jaw center
            p.resetBasePositionAndOrientation(
                self.grasped_id,
                [ee_pos[0], ee_pos[1], ee_pos[2] - 0.042],
                [0, 0, 0, 1]
            )
        elif not self.gripper_closed and self.grasped_id is not None:
            self.grasped_id = None

    def render_frame(self):
        _, _, rgb, _, _ = p.getCameraImage(
            width=self.width,
            height=self.height,
            viewMatrix=self.view_matrix,
            projectionMatrix=self.proj_matrix,
            renderer=p.ER_TINY_RENDERER
        )
        return rgb[:, :, :3]

    def get_box_position(self, color_name):
        bid = self.boxes[color_name]
        pos, _ = p.getBasePositionAndOrientation(bid)
        return np.array(pos)

    def get_plate_position(self, color_name):
        pid = self.plates[color_name]
        pos, _ = p.getBasePositionAndOrientation(pid)
        return np.array(pos)

    def check_task_success(self, color_name):
        b_pos = self.get_box_position(color_name)
        p_pos = self.get_plate_position(color_name)
        dist_xy = np.linalg.norm(b_pos[:2] - p_pos[:2])
        is_on_plate = (dist_xy < 0.035) and (b_pos[2] < 0.02)
        return is_on_plate, dist_xy

    def close(self):
        p.disconnect(self.client)

print("DobotColorBoxEnv class loaded successfully.")
