"""
dobot_realistic_table_env.py

PyBullet Simulation Environment with realistic tabletop objects:
- Hardcover Book (rectangular geometry, cover & spine)
- Ceramic Mug (cylindrical geometry with handle grasp zone)
- Desktop Storage Tray / Bin
- Multi-camera rendering:
  * Front 45-degree Third-Person View (640x480)
  * Wrist-Mounted In-Hand POV Camera (256x256)
  * Orthographic Overhead Side View (256x256)
"""

import os
import sys
import numpy as np
import pybullet as p
import pybullet_data
import cv2

from dobot_kinematics import DobotKinematics

GRIPPER_HEIGHT_OFFSET = 0.090  # 90mm from flange to finger tips


class DobotRealisticTableEnv:
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

        # Front 45-degree camera
        self.cam_target = [0.20, 0.0, -0.04]
        self.cam_eye = [0.42, 0.0, 0.32]
        self.cam_up = [0.0, 0.0, 1.0]
        self.fov = 45.0

        self.view_matrix = p.computeViewMatrix(self.cam_eye, self.cam_target, self.cam_up)
        self.proj_matrix = p.computeProjectionMatrixFOV(self.fov, 1.0, 0.05, 2.5)

        self.robot_id = None
        self.plane_id = None
        self.objects = {}
        self.grasped_id = None
        self.gripper_closed = False

        self.ee_link_index = 6
        self.gripper_j_left = 7
        self.gripper_j_right = 8

        self._load_scene()

    def _load_scene(self):
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)

        # Table surface (wood finish)
        col_t = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.02])
        vis_t = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.02], rgbaColor=[0.38, 0.30, 0.22, 1.0])
        self.plane_id = p.createMultiBody(0, col_t, vis_t, [0.20, 0.0, -0.07])

        # Load Dobot Magician URDF
        urdf_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "dobot_magician_ros2", "ai_training", "env", "dobot_pybullet.urdf")
        )
        self.robot_id = p.loadURDF(urdf_path, [0, 0, 0], useFixedBase=True)

        for j in [self.gripper_j_left, self.gripper_j_right]:
            p.changeDynamics(self.robot_id, j, lateralFriction=3.5, spinningFriction=0.2)

    def spawn_realistic_scene(self, book_pos=(0.23, 0.08), mug_pos=(0.20, 0.0), tray_pos=(0.22, -0.10)):
        """
        Spawns:
        1. Blue Hardcover Book at book_pos
        2. Red Ceramic Mug at mug_pos (distractor)
        3. Green Sorting Tray at tray_pos (destination)
        """
        table_top_z = -0.05

        # 1. Blue Hardcover Book: Rectangular flat slab (12cm x 9cm x 2cm)
        book_half = [0.050, 0.035, 0.010]
        col_book = p.createCollisionShape(p.GEOM_BOX, halfExtents=book_half)
        vis_book = p.createVisualShape(p.GEOM_BOX, halfExtents=book_half, rgbaColor=[0.15, 0.25, 0.85, 1.0])
        self.objects["book"] = p.createMultiBody(
            0.12, col_book, vis_book, [book_pos[0], book_pos[1], table_top_z + book_half[2]]
        )

        # 2. Red Mug: Cylinder (diameter 6cm, height 6cm)
        col_mug = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.030, height=0.060)
        vis_mug = p.createVisualShape(p.GEOM_CYLINDER, radius=0.030, length=0.060, rgbaColor=[0.85, 0.15, 0.15, 1.0])
        self.objects["mug"] = p.createMultiBody(
            0.20, col_mug, vis_mug, [mug_pos[0], mug_pos[1], table_top_z + 0.030]
        )

        # 3. Green Organizer Tray: Flat container (14cm x 14cm)
        col_tray = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.065, 0.065, 0.005])
        vis_tray = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.065, 0.065, 0.005], rgbaColor=[0.20, 0.80, 0.30, 0.85])
        self.objects["tray"] = p.createMultiBody(
            0.0, col_tray, vis_tray, [tray_pos[0], tray_pos[1], table_top_z + 0.005]
        )

        for _ in range(50):
            p.stepSimulation()

    def set_cartesian_target(self, x, y, z_tip, r=0.0, gripper_close=False):
        """Sets Dobot end-effector target through inverse kinematics."""
        self.gripper_closed = gripper_close
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

        # Gripper jaws
        jaw_pos = 0.0 if gripper_close else 0.0135
        p.resetJointState(self.robot_id, self.gripper_j_left, jaw_pos)
        p.resetJointState(self.robot_id, self.gripper_j_right, -jaw_pos)

        for _ in range(12):
            p.stepSimulation()
            self._update_gripper_physics()
        return True

    def _update_gripper_physics(self):
        ee_pos = p.getLinkState(self.robot_id, self.ee_link_index)[0]
        tip_pos = [ee_pos[0], ee_pos[1], ee_pos[2] - GRIPPER_HEIGHT_OFFSET]

        if self.gripper_closed and self.grasped_id is None:
            for name, oid in self.objects.items():
                if name == "tray":
                    continue
                o_pos, _ = p.getBasePositionAndOrientation(oid)
                dist = np.linalg.norm(np.array(tip_pos) - np.array(o_pos))
                if dist < 0.040:
                    self.grasped_id = oid
                    break

        if self.gripper_closed and self.grasped_id is not None:
            p.resetBasePositionAndOrientation(
                self.grasped_id,
                [tip_pos[0], tip_pos[1], tip_pos[2] + 0.005],
                [0, 0, 0, 1]
            )
        elif not self.gripper_closed and self.grasped_id is not None:
            self.grasped_id = None

    def render_frame(self):
        """Front 45-degree agent view."""
        _, _, rgb, _, _ = p.getCameraImage(
            width=self.width,
            height=self.height,
            viewMatrix=self.view_matrix,
            projectionMatrix=self.proj_matrix,
            renderer=p.ER_TINY_RENDERER
        )
        return rgb[:, :, :3]

    def render_wrist_frame(self, w=256, h=256):
        """Wrist-mounted tool camera."""
        ee_pos, ee_orn = p.getLinkState(self.robot_id, self.ee_link_index)[:2]
        rot_mat = p.getMatrixFromQuaternion(ee_orn)
        fwd = [rot_mat[2], rot_mat[5], rot_mat[8]]
        up = [-rot_mat[1], -rot_mat[4], -rot_mat[7]]

        cam_eye = [ee_pos[0], ee_pos[1], ee_pos[2] - 0.02]
        cam_target = [cam_eye[0] - fwd[0]*0.2, cam_eye[1] - fwd[1]*0.2, cam_eye[2] - fwd[2]*0.2]

        vm = p.computeViewMatrix(cam_eye, cam_target, up)
        pm = p.computeProjectionMatrixFOV(65.0, 1.0, 0.02, 1.0)
        _, _, rgb, _, _ = p.getCameraImage(w, h, vm, pm, renderer=p.ER_TINY_RENDERER)
        return rgb[:, :, :3]
