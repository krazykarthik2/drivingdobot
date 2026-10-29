import os
import json
import numpy as np
import cv2
import torch
from dobot_kinematics import DobotKinematics
from system0a_base_controller import System0ABaseController
from system1_perception import System1PerceptionEngine, CameraProjector
from system2_reasoning import System2ReasoningEngine

class DobotResidualSimEnv:
    """
    Simulated 4-DOF Dobot Environment tailored for Gripper Residual / Delta Control.
    Calibrated strictly for Gripper End-Effector (not sucker):
      - GRIPPER_TABLE_Z = 0.070 m (perfect vertical engagement with table object)
      - STANDBY_Z = 0.150 m (high clearance baseline)
      - OBSTACLE ARC HEIGHT = +0.120 m (soars completely over red obstacle cylinders)
    """
    def __init__(self, table_z=0.070, standby_z=0.150):
        self.kin = DobotKinematics()
        self.base_solver = System0ABaseController(table_z=table_z, standby_z=standby_z)
        self.perception = System1PerceptionEngine(width=128, height=128, table_z=table_z)
        self.projector = CameraProjector(width=640, height=480, table_z=table_z)
        self.reasoning = System2ReasoningEngine()
        
        self.table_z = table_z
        self.standby_z = standby_z
        self.reset()

    def reset(self, prompt=None, add_obstacle=True, is_weird_shape=True):
        pick_angle = float(np.random.uniform(-70.0, 70.0)) if is_weird_shape else 0.0
        pick_type = "wrench" if is_weird_shape else "cube"
        pick_color = "green"
        
        # Tabletop layout: Pick on one side, Place on other side, with realistic 16-20cm separation
        pick_rad = np.random.uniform(0.20, 0.25)
        pick_th = np.random.uniform(0.35, 0.60) # left side
        self.pick_pos = np.array([pick_rad * np.cos(pick_th), pick_rad * np.sin(pick_th), self.table_z], dtype=np.float32)

        # Place target on the right side
        place_color = "blue"
        place_type = "platform"
        place_rad = np.random.uniform(0.20, 0.25)
        place_th = np.random.uniform(-0.60, -0.35) # right side
        self.place_pos = np.array([place_rad * np.cos(place_th), place_rad * np.sin(place_th), self.table_z], dtype=np.float32)

        # Obstacle placed at midpoint
        self.obstacles = []
        if add_obstacle:
            midpoint = (self.pick_pos + self.place_pos) / 2.0
            midpoint[2] = self.table_z
            self.obstacles.append({
                "color": "red",
                "type": "cup",
                "pos": midpoint,
                "angle": 0.0
            })

        self.world_objects = [
            {"color": pick_color, "type": pick_type, "pos": self.pick_pos, "angle": pick_angle, "is_wrench": (pick_type == "wrench")},
            {"color": place_color, "type": place_type, "pos": self.place_pos, "angle": 0.0, "is_wrench": False},
        ] + self.obstacles

        if prompt:
            self.instruction = prompt
        else:
            self.instruction = f"Pick up the {pick_color} wrench and deposit it into the {place_color} {place_type} avoiding the red obstacle."

        # Parse instruction via System 2
        self.reasoning_targets = self.reasoning.parse_instruction(self.instruction)

        # Ensure world objects reflect colors requested by user
        if "pick" in self.reasoning_targets and "color" in self.reasoning_targets["pick"]:
            self.world_objects[0]["color"] = self.reasoning_targets["pick"]["color"]
        if "place" in self.reasoning_targets and "color" in self.reasoning_targets["place"]:
            self.world_objects[1]["color"] = self.reasoning_targets["place"]["color"]

        # Generate zero-background silhouette image via System 1
        self.silhouette_128, self.silhouette_full, self.meta = self.perception.segment_scene_to_silhouettes(
            None, self.world_objects, self.reasoning_targets
        )

        self.ee_pos = np.array([0.20, 0.0, self.standby_z, 0.0], dtype=np.float32)
        return self.silhouette_128, self.instruction

    def generate_expert_demonstration(self, num_steps=30):
        """
        Phase 2 Shadow Data Collection:
        Computes both the Naive Programmed trajectory and the Expert Corrected trajectory.
        Calculates the exact corrective Delta: Delta = Expert - Naive
        """
        naive_cartesian = []
        naive_joints = []
        
        start_pt = self.ee_pos[:3]
        pick_pt = self.pick_pos.copy()
        place_pt = self.place_pos.copy()

        half_steps = num_steps // 2

        # 1. Naive Path (Straight-line horizontally, flat yaw=0, crashes right through obstacle)
        for i in range(half_steps):
            a = i / float(half_steps - 1)
            pt = (1 - a) * start_pt + a * pick_pt
            j = self.kin.inverse(pt[0], pt[1], pt[2], 0.0)
            naive_cartesian.append(np.array([pt[0], pt[1], pt[2], 0.0], dtype=np.float32))
            naive_joints.append(j)

        for i in range(half_steps):
            a = i / float(half_steps - 1)
            pt = (1 - a) * pick_pt + a * place_pt
            j = self.kin.inverse(pt[0], pt[1], pt[2], 0.0)
            naive_cartesian.append(np.array([pt[0], pt[1], pt[2], 0.0], dtype=np.float32))
            naive_joints.append(j)

        naive_joints = np.array(naive_joints, dtype=np.float32)

        # 2. Expert Path (Twists wrist to align with wrench angle, and arcs HIGH over red obstacle)
        pick_angle_rad = np.radians(self.world_objects[0]["angle"])
        expert_cartesian = []
        expert_joints = []

        # Reach with wrist alignment
        for i in range(half_steps):
            a = i / float(half_steps - 1)
            pt = (1 - a) * start_pt + a * pick_pt
            expert_yaw = a * pick_angle_rad
            j = self.kin.inverse(pt[0], pt[1], pt[2], expert_yaw)
            expert_cartesian.append(np.array([pt[0], pt[1], pt[2], expert_yaw], dtype=np.float32))
            expert_joints.append(j)

        # Transfer with obstacle clearance (High Arc in Z)
        has_obs = len(self.obstacles) > 0
        for i in range(half_steps):
            a = i / float(half_steps - 1)
            pt = (1 - a) * pick_pt + a * place_pt
            if has_obs:
                # Strong parabolic Z clearance: lifts +0.100m in air to completely clear obstacle
                arc_z = 0.100 * np.sin(np.pi * a)
                pt[2] += arc_z
            expert_yaw = (1 - a) * pick_angle_rad
            j = self.kin.inverse(pt[0], pt[1], pt[2], expert_yaw)
            expert_cartesian.append(np.array([pt[0], pt[1], pt[2], expert_yaw], dtype=np.float32))
            expert_joints.append(j)

        expert_joints = np.array(expert_joints, dtype=np.float32)
        delta_joints = expert_joints - naive_joints

        return {
            "silhouette_128": self.silhouette_128,
            "instruction": self.instruction,
            "naive_joints": naive_joints,
            "expert_joints": expert_joints,
            "delta_joints": delta_joints,
            "has_obstacle": has_obs,
            "pick_angle_deg": self.world_objects[0]["angle"]
        }
