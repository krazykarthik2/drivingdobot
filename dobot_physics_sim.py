"""
PyBullet Physical Simulation Environment for Dobot Magician 4-DOF.
Authentic Solid Geometries (No rolling cylinders, no drifting bodies):
1. Pick Target: Authentic Rigid Wrench
   - Rectangular slotted head + elongated flat handle bar.
   - High lateral friction (2.5) + damping to prevent physical sliding/drifting.
2. Place Target: Rectangular Bin / Tray with 4 outer lip rim walls
   - Object drops into the cavity and rests firmly inside the receptacle.
3. Avoid Obstacle: Stable Red Mug / Tall Barrier with Flat Base and Side Handle
   - Heavy base (0.35kg), high friction, completely stable without tipping or rolling.
4. Centered Front 45-Degree Camera:
   - Positioned directly in front of Dobot (Y=0.0) with exact 45-degree downward elevation.
"""

import os
import time
import numpy as np
import pybullet as p
import pybullet_data
import cv2

from dobot_kinematics import DobotKinematics

class DobotPybulletPhysicsSim:
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

        # Centered Front 45-Degree Camera Setup
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
        self.object_ids = {}
        self.grasp_constraint_id = None
        self.gripper_closed = False
        self.grasped = False
        self.load_environment()

    def load_environment(self):
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)

        # Table & Floor Plane
        self.plane_id = p.loadURDF("plane.urdf", [0, 0, -0.05])
        
        # Load Dobot URDF
        urdf_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "dobot_magician_ros2", "ai_training", "env", "dobot_pybullet.urdf")
        )
        self.robot_id = p.loadURDF(urdf_path, [0, 0, 0], useFixedBase=True)

        self.arm_joints = [1, 2, 3, 6]
        self.gripper_j_left = 7
        self.gripper_j_right = 8
        self.ee_link_index = 6

        # Dynamics
        p.changeDynamics(self.robot_id, self.gripper_j_left, lateralFriction=3.0, spinningFriction=0.2)
        p.changeDynamics(self.robot_id, self.gripper_j_right, lateralFriction=3.0, spinningFriction=0.2)
        p.changeDynamics(self.plane_id, -1, lateralFriction=1.5)

    def spawn_scene_objects(self, pick_pos, pick_angle_deg, place_pos, obs_pos=None, pick_type="wrench"):
        """
        Spawns realistic physical non-rolling collision bodies.
        """
        if self.grasp_constraint_id is not None:
            try:
                p.removeConstraint(self.grasp_constraint_id)
            except Exception:
                pass
            self.grasp_constraint_id = None

        for oid in self.object_ids.values():
            try:
                p.removeBody(oid)
            except Exception:
                pass
        self.object_ids = {}
        self.grasped = False
        self.gripper_closed = False

        # 1. Authentic Pick Wrench (slotted head + flat handle)
        pick_orn = p.getQuaternionFromEuler([0, 0, np.radians(pick_angle_deg)])
        
        col_shaft = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.040, 0.007, 0.008])
        vis_shaft = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.040, 0.007, 0.008], rgbaColor=[0.15, 0.85, 0.25, 1.0])
        col_head = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.015, 0.018, 0.008])
        vis_head = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.015, 0.018, 0.008], rgbaColor=[0.15, 0.85, 0.25, 1.0])

        self.object_ids["pick"] = p.createMultiBody(
            baseMass=0.08,
            baseCollisionShapeIndex=col_shaft,
            baseVisualShapeIndex=vis_shaft,
            basePosition=[pick_pos[0], pick_pos[1], -0.042],
            baseOrientation=pick_orn,
            linkMasses=[0.03],
            linkCollisionShapeIndices=[col_head],
            linkVisualShapeIndices=[vis_head],
            linkPositions=[[0.048, 0.0, 0.0]],
            linkOrientations=[[0, 0, 0, 1]],
            linkInertialFramePositions=[[0, 0, 0]],
            linkInertialFrameOrientations=[[0, 0, 0, 1]],
            linkParentIndices=[0],
            linkJointTypes=[p.JOINT_FIXED],
            linkJointAxis=[[0, 0, 0]]
        )
        p.changeDynamics(self.object_ids["pick"], -1, lateralFriction=2.8, linearDamping=0.6, angularDamping=0.9)
        p.changeDynamics(self.object_ids["pick"], 0, lateralFriction=2.8, linearDamping=0.6, angularDamping=0.9)

        # 2. Place Receptacle: Blue Bin / Tray with 4 walls
        col_base = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.035, 0.035, 0.004])
        vis_base = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.035, 0.035, 0.004], rgbaColor=[0.15, 0.40, 0.90, 1.0])
        col_wall_x = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.035, 0.004, 0.012])
        vis_wall_x = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.035, 0.004, 0.012], rgbaColor=[0.12, 0.35, 0.80, 1.0])
        col_wall_y = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.004, 0.035, 0.012])
        vis_wall_y = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.004, 0.035, 0.012], rgbaColor=[0.12, 0.35, 0.80, 1.0])

        self.object_ids["place"] = p.createMultiBody(
            baseMass=0.0, # static table receptacle
            baseCollisionShapeIndex=col_base,
            baseVisualShapeIndex=vis_base,
            basePosition=[place_pos[0], place_pos[1], -0.046],
            baseOrientation=[0, 0, 0, 1],
            linkMasses=[0.0, 0.0, 0.0, 0.0],
            linkCollisionShapeIndices=[col_wall_x, col_wall_x, col_wall_y, col_wall_y],
            linkVisualShapeIndices=[vis_wall_x, vis_wall_x, vis_wall_y, vis_wall_y],
            linkPositions=[[0, 0.035, 0.012], [0, -0.035, 0.012], [0.035, 0, 0.012], [-0.035, 0, 0.012]],
            linkOrientations=[[0,0,0,1], [0,0,0,1], [0,0,0,1], [0,0,0,1]],
            linkInertialFramePositions=[[0,0,0], [0,0,0], [0,0,0], [0,0,0]],
            linkInertialFrameOrientations=[[0,0,0,1], [0,0,0,1], [0,0,0,1], [0,0,0,1]],
            linkParentIndices=[0, 0, 0, 0],
            linkJointTypes=[p.JOINT_FIXED, p.JOINT_FIXED, p.JOINT_FIXED, p.JOINT_FIXED],
            linkJointAxis=[[0,0,0], [0,0,0], [0,0,0], [0,0,0]]
        )

        # 3. Stable Red Obstacle (Cup / Mug with flat non-rolling base & handle)
        if obs_pos is not None:
            col_mug = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.022, 0.022, 0.035])
            vis_mug = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.022, 0.022, 0.035], rgbaColor=[0.90, 0.15, 0.15, 1.0])
            col_m_handle = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.007, 0.005, 0.020])
            vis_m_handle = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.007, 0.005, 0.020], rgbaColor=[0.75, 0.10, 0.10, 1.0])

            self.object_ids["obstacle"] = p.createMultiBody(
                baseMass=0.35, # heavy and totally stable
                baseCollisionShapeIndex=col_mug,
                baseVisualShapeIndex=vis_mug,
                basePosition=[obs_pos[0], obs_pos[1], -0.015],
                baseOrientation=[0, 0, 0, 1],
                linkMasses=[0.04],
                linkCollisionShapeIndices=[col_m_handle],
                linkVisualShapeIndices=[vis_m_handle],
                linkPositions=[[0.026, 0.0, 0.0]],
                linkOrientations=[[0,0,0,1]],
                linkInertialFramePositions=[[0,0,0]],
                linkInertialFrameOrientations=[[0,0,0,1]],
                linkParentIndices=[0],
                linkJointTypes=[p.JOINT_FIXED],
                linkJointAxis=[[0,0,0]]
            )
            p.changeDynamics(self.object_ids["obstacle"], -1, lateralFriction=3.0, linearDamping=0.8, angularDamping=0.9)

        # Settle simulation
        for _ in range(40):
            p.stepSimulation()

    def set_dobot_joints(self, j1, j2, j3, j4, gripper_close=False):
        """
        Sets arm joint states and drives gripper physics with grasp lock.
        """
        j2_urdf = np.pi / 2.0 - j2
        j3_urdf = -np.pi / 2.0 - j3

        p.resetJointState(self.robot_id, 1, j1)
        p.resetJointState(self.robot_id, 2, j2_urdf)
        p.resetJointState(self.robot_id, 3, j3_urdf)
        p.resetJointState(self.robot_id, 4, -j2_urdf)
        p.resetJointState(self.robot_id, 5, -j3_urdf)
        p.resetJointState(self.robot_id, 6, j4)

        # Gripper physical actuation
        jaw_pos = 0.012 if gripper_close else 0.0
        p.setJointMotorControl2(self.robot_id, self.gripper_j_left, p.POSITION_CONTROL, targetPosition=jaw_pos, force=100)
        p.setJointMotorControl2(self.robot_id, self.gripper_j_right, p.POSITION_CONTROL, targetPosition=-jaw_pos, force=100)

        self.gripper_closed = gripper_close
        self._update_gripper_physics()
        p.stepSimulation()

    def _update_gripper_physics(self):
        if "pick" not in self.object_ids:
            return

        pick_id = self.object_ids["pick"]

        if self.gripper_closed and not self.grasped:
            contact_l = p.getContactPoints(bodyA=self.robot_id, bodyB=pick_id, linkIndexA=self.gripper_j_left)
            contact_r = p.getContactPoints(bodyA=self.robot_id, bodyB=pick_id, linkIndexA=self.gripper_j_right)
            
            ee_state = p.getLinkState(self.robot_id, self.ee_link_index)
            ee_pos = np.array(ee_state[0])
            pick_pos, _ = p.getBasePositionAndOrientation(pick_id)
            dist_ee_pick = np.linalg.norm(ee_pos - np.array(pick_pos))

            if len(contact_l) > 0 or len(contact_r) > 0 or dist_ee_pick < 0.065:
                self.grasp_constraint_id = p.createConstraint(
                    parentBodyUniqueId=self.robot_id,
                    parentLinkIndex=self.ee_link_index,
                    childBodyUniqueId=pick_id,
                    childLinkIndex=-1,
                    jointType=p.JOINT_FIXED,
                    jointAxis=[0, 0, 0],
                    parentFramePosition=[0, 0, -0.02],
                    childFramePosition=[0, 0, 0]
                )
                self.grasped = True

        elif not self.gripper_closed and self.grasped:
            if self.grasp_constraint_id is not None:
                p.removeConstraint(self.grasp_constraint_id)
                self.grasp_constraint_id = None
            self.grasped = False

    def check_obstacle_collision(self):
        if "obstacle" not in self.object_ids:
            return False
        obs_id = self.object_ids["obstacle"]
        arm_contacts = p.getContactPoints(bodyA=self.robot_id, bodyB=obs_id)
        obj_contacts = []
        if self.grasped and "pick" in self.object_ids:
            obj_contacts = p.getContactPoints(bodyA=self.object_ids["pick"], bodyB=obs_id)
            
        return len(arm_contacts) > 0 or len(obj_contacts) > 0

    def render_front_45deg_camera(self):
        _, _, rgb_raw, _, _ = p.getCameraImage(
            width=self.width,
            height=self.height,
            viewMatrix=self.view_matrix,
            projectionMatrix=self.proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
        rgb = np.array(rgb_raw, dtype=np.uint8).reshape((self.height, self.width, 4))[:, :, :3]
        return rgb

    def disconnect(self):
        try:
            p.disconnect(self.client)
        except Exception:
            pass

if __name__ == "__main__":
    sim = DobotPybulletPhysicsSim(render_gui=False)
    sim.spawn_scene_objects(
        pick_pos=np.array([0.22, 0.08, -0.042]),
        pick_angle_deg=35.0,
        place_pos=np.array([0.22, -0.08, -0.046]),
        obs_pos=np.array([0.22, 0.0, -0.015]),
        pick_type="wrench"
    )
    rgb = sim.render_front_45deg_camera()
    cv2.imwrite("test_authentic_objects.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    print("Saved test_authentic_objects.png with realistic wrench, tray receptacle, and stable mug obstacle!")
    sim.disconnect()
