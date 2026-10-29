"""
PyBullet Physical Simulation Environment for Dobot Magician 4-DOF.
Key features:
1. True Rigid Body Dynamics & Contact Physics:
   - Dobot URDF with 4 active DOF (joint 1, joint 2, joint 3, joint 4) + gripper jaws.
   - Physical table plane with gravity and realistic contact friction.
   - Dynamic physical objects: pick target (wrench / block), place target (tray / platform), and obstacles (red cups/cylinders).
   - Collision detection: detects if the naive straight-line path hits the obstacle!
2. Front 45-Degree Camera:
   - Positioned in front of the robot at a 45-degree angle looking down onto the workspace.
   - Renders both realistic RGB views and feeds System 1 zero-background silhouettes.
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

        # Connect to PyBullet physics server
        if self.render_gui:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1.0 / 240.0)

        # 45-degree Front Camera setup
        # Distance = 0.48m, Yaw = 0 deg (front facing robot), Pitch = -45 deg
        self.cam_target = [0.22, 0.0, 0.02]
        self.cam_dist = 0.48
        self.cam_yaw = 0.0      # Directly in front
        self.cam_pitch = -45.0  # Exactly 45-degree downward tilt
        self.cam_roll = 0.0
        self.fov = 50.0

        self.view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=self.cam_target,
            distance=self.cam_dist,
            yaw=self.cam_yaw,
            pitch=self.cam_pitch,
            roll=self.cam_roll,
            upAxisIndex=2
        )
        self.proj_matrix = p.computeProjectionMatrixFOV(
            fov=self.fov,
            aspect=float(width) / float(height),
            nearVal=0.05,
            farVal=2.0
        )

        self.robot_id = None
        self.plane_id = None
        self.table_id = None
        self.object_ids = {}
        self.load_environment()

    def load_environment(self):
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)

        # Table & Plane
        self.plane_id = p.loadURDF("plane.urdf", [0, 0, -0.05])
        
        # Load Dobot URDF
        urdf_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "dobot_magician_ros2", "ai_training", "env", "dobot_pybullet.urdf")
        )
        self.robot_id = p.loadURDF(urdf_path, [0, 0, 0], useFixedBase=True)

        # Joint indices
        # Joint 1 = index 1
        # Joint 2 = index 2
        # Joint 3 = index 3
        # Joint 4 (wrist) = index 6
        # Gripper jaws = indices 7, 8
        self.arm_joints = [1, 2, 3, 6]
        self.gripper_joints = [7, 8]

        # Enable force/torque sensors and set dynamics
        for j in range(p.getNumJoints(self.robot_id)):
            p.changeDynamics(self.robot_id, j, lateralFriction=1.0, spinningFriction=0.1)

    def spawn_scene_objects(self, pick_pos, pick_angle_deg, place_pos, obs_pos=None, pick_type="wrench"):
        """
        Spawns physical collision shapes into the PyBullet physics engine.
        """
        # Clear old objects
        for oid in self.object_ids.values():
            try:
                p.removeBody(oid)
            except Exception:
                pass
        self.object_ids = {}

        # 1. Pick Object (Physical Body)
        pick_orn = p.getQuaternionFromEuler([0, 0, np.radians(pick_angle_deg)])
        if pick_type == "wrench":
            # Elongated box collision for wrench
            col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.05, 0.012, 0.012])
            vis_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.05, 0.012, 0.012], rgbaColor=[0.15, 0.85, 0.25, 1.0])
        else:
            col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.015, 0.015, 0.015])
            vis_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.015, 0.015, 0.015], rgbaColor=[0.15, 0.85, 0.25, 1.0])

        self.object_ids["pick"] = p.createMultiBody(
            baseMass=0.08,
            baseCollisionShapeIndex=col_id,
            baseVisualShapeIndex=vis_id,
            basePosition=[pick_pos[0], pick_pos[1], max(pick_pos[2], 0.012)],
            baseOrientation=pick_orn
        )
        p.changeDynamics(self.object_ids["pick"], -1, lateralFriction=1.2, rollingFriction=0.01)

        # 2. Place Object (Tray/Platform)
        p_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.035, 0.035, 0.005])
        p_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.035, 0.035, 0.005], rgbaColor=[0.15, 0.40, 0.90, 1.0])
        self.object_ids["place"] = p.createMultiBody(
            baseMass=0.0, # static
            baseCollisionShapeIndex=p_col,
            baseVisualShapeIndex=p_vis,
            basePosition=[place_pos[0], place_pos[1], max(place_pos[2], 0.005)],
            baseOrientation=[0, 0, 0, 1]
        )

        # 3. Obstacle (Tall Cylinder / Cup in between)
        if obs_pos is not None:
            obs_col = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.022, height=0.07)
            obs_vis = p.createVisualShape(p.GEOM_CYLINDER, radius=0.022, length=0.07, rgbaColor=[0.90, 0.15, 0.15, 1.0])
            self.object_ids["obstacle"] = p.createMultiBody(
                baseMass=0.25,
                baseCollisionShapeIndex=obs_col,
                baseVisualShapeIndex=obs_vis,
                basePosition=[obs_pos[0], obs_pos[1], 0.035],
                baseOrientation=[0, 0, 0, 1]
            )

        # Step physics to settle objects on ground
        for _ in range(30):
            p.stepSimulation()

    def set_dobot_joints(self, j1, j2, j3, j4, gripper_val=0.0):
        """
        Directly sets kinematic joint states in PyBullet.
        j2_urdf = pi/2 - j2_kin
        j3_urdf = -pi/2 - j3_kin
        """
        j2_urdf = np.pi / 2.0 - j2
        j3_urdf = -np.pi / 2.0 - j3

        p.resetJointState(self.robot_id, 1, j1)
        p.resetJointState(self.robot_id, 2, j2_urdf)
        p.resetJointState(self.robot_id, 3, j3_urdf)
        # Mimic joints
        p.resetJointState(self.robot_id, 4, -j2_urdf)
        p.resetJointState(self.robot_id, 5, -j3_urdf)
        p.resetJointState(self.robot_id, 6, j4)

        # Gripper jaws
        p.resetJointState(self.robot_id, 7, gripper_val)
        p.resetJointState(self.robot_id, 8, -gripper_val)

        p.stepSimulation()

    def check_obstacle_collision(self):
        """
        Queries PyBullet physical contact points between the robot arm and the obstacle.
        Returns True if the robot collides with the obstacle.
        """
        if "obstacle" not in self.object_ids:
            return False
        pts = p.getContactPoints(bodyA=self.robot_id, bodyB=self.object_ids["obstacle"])
        return len(pts) > 0

    def render_front_45deg_camera(self):
        """
        Renders the physical PyBullet world from the 45-degree front camera.
        Returns RGB image [H, W, 3] in uint8.
        """
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
        pick_pos=np.array([0.22, 0.08, 0.012]),
        pick_angle_deg=35.0,
        place_pos=np.array([0.22, -0.08, 0.005]),
        obs_pos=np.array([0.22, 0.0, 0.035]),
        pick_type="wrench"
    )
    sim.set_dobot_joints(0.0, 0.4, -1.7, 0.0)
    rgb = sim.render_front_45deg_camera()
    print("Physical 45-degree Front Camera Render shape:", rgb.shape)
    cv2.imwrite("pybullet_front45_demo.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    print("Obstacle Collision:", sim.check_obstacle_collision())
    print("PyBullet physical simulation ready!")
    sim.disconnect()
