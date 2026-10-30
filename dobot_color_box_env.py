"""
Dobot Magician Multi-Color Box Pick-and-Place Benchmark Environment.
Replaces Franka with the 4-DOF Dobot Magician robot and evaluates sorting
small colored blocks (Red, Green, Blue, Yellow) into corresponding colored receptacles/plates.
Produces complete HD execution videos with closed-loop servo precision,
accounting for the 90mm tall end-effector gripper offset and proper gripper state signaling.
Includes point cloud extraction for 3D Diffusion Policy (DP3).
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

# Gripper Height Offset: 90 mm from tool flange / ee_link to fingertip grasp center
GRIPPER_HEIGHT_OFFSET = 0.090  # 90mm

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

        # Calibrated Front 45-degree elevated camera matching standard manipulation benchmarks
        self.cam_target = [0.20, 0.0, -0.04]
        self.cam_eye = [0.42, 0.0, 0.32]
        self.cam_up = [0.0, 0.0, 1.0]
        self.fov = 45.0

        self.view_matrix = p.computeViewMatrix(
            cameraEyePosition=self.cam_eye,
            cameraTargetPosition=self.cam_target,
            cameraUpVector=self.cam_up
        )
        self.proj_matrix = p.computeProjectionMatrixFOV(
            fov=self.fov,
            aspect=1.0,
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

        # Natural table surface matching robot tabletop domains
        col_t = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.02])
        vis_t = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.02], rgbaColor=[0.32, 0.25, 0.20, 1.0])
        self.plane_id = p.createMultiBody(0, col_t, vis_t, [0.20, 0.0, -0.07])

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

    def set_cartesian_target(self, x, y, z_tip, r=0.0, gripper_close=False):
        """
        Sets end-effector fingertip Cartesian target (x, y, z_tip).
        Accounts for the 90mm gripper height offset automatically by solving IK
        for tool flange z_ee = z_tip + GRIPPER_HEIGHT_OFFSET.
        """
        z_ee = z_tip + GRIPPER_HEIGHT_OFFSET
        ik_sol = self.kin.inverse(x, y, z_ee, yaw=r)
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

        jaw_pos = 0.0 if gripper_close else 0.0135
        p.resetJointState(self.robot_id, self.gripper_j_left, jaw_pos)
        p.resetJointState(self.robot_id, self.gripper_j_right, -jaw_pos)

        self.gripper_closed = gripper_close
        self._update_gripper_physics()
        p.stepSimulation()
        return True

    def get_gripper_state(self):
        """
        Returns the physically measured gripper state and string.
        Output state: 'GRIPPED' when closed around object, 'OPEN' when retracted open.
        """
        jaw_left = p.getJointState(self.robot_id, self.gripper_j_left)[0]
        is_gripped = (jaw_left < 0.005) and self.gripper_closed
        return {
            "state_str": "GRIPPED" if is_gripped else "OPEN",
            "is_gripped": is_gripped,
            "jaw_left_pos": jaw_left,
            "grasped_object": self.grasped_id
        }

    def _update_gripper_physics(self):
        ee_pos = p.getLinkState(self.robot_id, self.ee_link_index)[0]
        tip_pos = [ee_pos[0], ee_pos[1], ee_pos[2] - GRIPPER_HEIGHT_OFFSET]

        if self.gripper_closed and self.grasped_id is None:
            for cname, bid in self.boxes.items():
                b_pos, _ = p.getBasePositionAndOrientation(bid)
                dist = np.linalg.norm(np.array(tip_pos) - np.array(b_pos))
                if dist < 0.035:
                    self.grasped_id = bid
                    break

        if self.gripper_closed and self.grasped_id is not None:
            p.resetBasePositionAndOrientation(
                self.grasped_id,
                [tip_pos[0], tip_pos[1], tip_pos[2] + 0.005],
                [0, 0, 0, 1]
            )
        elif not self.gripper_closed and self.grasped_id is not None:
            self.grasped_id = None

    def get_robot_state_9d(self):
        """
        Returns 9-dim proprioceptive state matching DP3 agent_pos:
        [ee_pos (3), finger_right (3), finger_left (3)]
        """
        ee_pos = p.getLinkState(self.robot_id, self.ee_link_index)[0]
        tip_pos = [ee_pos[0], ee_pos[1], ee_pos[2] - GRIPPER_HEIGHT_OFFSET]
        f_left = p.getLinkState(self.robot_id, self.gripper_j_left)[0]
        f_right = p.getLinkState(self.robot_id, self.gripper_j_right)[0]
        return np.concatenate([tip_pos, f_right, f_left]).astype(np.float32)

    def get_point_cloud(self, num_points=512):
        """
        Extracts 3D Point Cloud (Nx3) from the camera depth buffer without knowing box ground-truth positions.
        """
        w, h = 128, 128
        _, _, rgb, depth, _ = p.getCameraImage(
            w, h, self.view_matrix, self.proj_matrix, renderer=p.ER_TINY_RENDERER
        )
        near, far = 0.05, 2.5
        z_ndc = 2.0 * depth - 1.0
        z_eye = 2.0 * near * far / (far + near - z_ndc * (far - near))

        xs = np.linspace(-1, 1, w)
        ys = np.linspace(1, -1, h)
        x_grid, y_grid = np.meshgrid(xs, ys)
        aspect = float(w) / float(h)
        tan_half_fov = np.tan(np.deg2rad(self.fov / 2.0))
        x_eye = x_grid * z_eye * tan_half_fov * aspect
        y_eye = y_grid * z_eye * tan_half_fov

        pts_cam = np.stack([x_eye, y_eye, -z_eye, np.ones_like(z_eye)], axis=-1).reshape(-1, 4)
        inv_view = np.linalg.inv(np.array(self.view_matrix).reshape(4, 4).T)
        pts_world = (pts_cam @ inv_view.T)[:, :3]

        # Filter out points below table/plane or too far
        mask = (pts_world[:, 2] > -0.05) & (pts_world[:, 2] < 0.25) & (pts_world[:, 0] > 0.05) & (pts_world[:, 0] < 0.40)
        pts_filtered = pts_world[mask]

        if len(pts_filtered) < num_points:
            pts_filtered = pts_world

        indices = np.random.choice(pts_filtered.shape[0], num_points, replace=(len(pts_filtered) < num_points))
        return pts_filtered[indices].astype(np.float32)

    def render_side_frame(self, width=256, height=256):
        """
        Renders secondary third-person perspective (camera3 / side view) at [0.24, -0.36, 0.32],
        providing lateral depth perception orthogonal to the front camera.
        """
        side_eye = [0.24, -0.36, 0.32]
        side_target = [0.20, 0.0, -0.04]
        side_up = [0.0, 0.0, 1.0]

        view_mat = p.computeViewMatrix(
            cameraEyePosition=side_eye,
            cameraTargetPosition=side_target,
            cameraUpVector=side_up
        )
        proj_mat = p.computeProjectionMatrixFOV(
            fov=45.0,
            aspect=float(width) / float(height),
            nearVal=0.05,
            farVal=2.5
        )
        _, _, rgb, _, _ = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_mat,
            projectionMatrix=proj_mat,
            renderer=p.ER_TINY_RENDERER
        )
        return rgb[:, :, :3]

    def render_wrist_frame(self, width=256, height=256):
        """
        Renders eye-in-hand wrist camera observation mounted directly on the gripper carriage.
        Tracks the end-effector in real-time, looking down through the fingertip pinch opening.
        """
        ee_pos, ee_orn = p.getLinkState(self.robot_id, self.ee_link_index)[:2]
        wrist_eye = np.array(ee_pos) + np.array([-0.05, 0.0, -0.02])
        wrist_target = np.array(ee_pos) + np.array([0.0, 0.0, -0.09])
        
        view_mat = p.computeViewMatrix(
            cameraEyePosition=wrist_eye.tolist(),
            cameraTargetPosition=wrist_target.tolist(),
            cameraUpVector=[0, 0, 1]
        )
        proj_mat = p.computeProjectionMatrixFOV(
            fov=70.0,
            aspect=float(width) / float(height),
            nearVal=0.01,
            farVal=1.5
        )
        _, _, rgb, _, _ = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_mat,
            projectionMatrix=proj_mat,
            renderer=p.ER_TINY_RENDERER
        )
        return rgb[:, :, :3]

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
