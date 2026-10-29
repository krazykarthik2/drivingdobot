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
    Simulated 4-DOF Dobot Environment tailored for Residual / Delta Control.
    Features:
      - 40-degree front elevated camera view
      - Random weird prompts (shapes: wrench, bar, cube, cup, bin, obstacle; varied colors)
      - Complex geometries: weirdly oriented objects requiring specific wrist twist (yaw)
      - Obstacles in the path requiring arcing Z clearances
      - Analytical Base Solver path generation vs Human / Expert path generation
      - Exact Delta recording: Delta = (Expert Plan - Naive Plan)
    """
    def __init__(self, table_z=-0.05, standby_z=0.10):
        self.kin = DobotKinematics()
        self.base_solver = System0ABaseController(table_z=table_z, standby_z=standby_z)
        self.perception = System1PerceptionEngine(width=128, height=128)
        self.projector = CameraProjector(width=640, height=480, table_z=table_z)
        self.reasoning = System2ReasoningEngine()
        
        self.table_z = table_z
        self.standby_z = standby_z
        self.reset()

    def reset(self, prompt=None, add_obstacle=True, is_weird_shape=True):
        """
        Resets environment with randomized objects, obstacles, and orientations.
        """
        # Random pick target
        pick_angle = float(np.random.uniform(-75.0, 75.0)) if is_weird_shape else 0.0
        pick_type = "wrench" if is_weird_shape else "cube"
        pick_color = np.random.choice(["green", "yellow", "purple", "cyan"])
        
        # Workspace limits for Dobot
        # R in [0.18, 0.28], angle in [-0.6, 0.6] rad
        pick_rad = np.random.uniform(0.18, 0.27)
        pick_th = np.random.uniform(-0.55, 0.55)
        self.pick_pos = np.array([pick_rad * np.cos(pick_th), pick_rad * np.sin(pick_th), self.table_z], dtype=np.float32)

        # Place target
        avail_place_colors = [c for c in ["blue", "orange", "cyan"] if c != pick_color]
        place_color = np.random.choice(avail_place_colors)
        place_type = np.random.choice(["bin", "platform", "tray"])
        
        # Ensure distance between pick and place
        for _ in range(50):
            place_rad = np.random.uniform(0.18, 0.27)
            place_th = np.random.uniform(-0.55, 0.55)
            pos_candidate = np.array([place_rad * np.cos(place_th), place_rad * np.sin(place_th), self.table_z], dtype=np.float32)
            if np.linalg.norm(pos_candidate[:2] - self.pick_pos[:2]) > 0.10:
                self.place_pos = pos_candidate
                break
        else:
            self.place_pos = np.array([0.22, -0.12, self.table_z], dtype=np.float32)

        # Obstacle setup (in-between pick and place)
        self.obstacles = []
        if add_obstacle:
            # Place obstacle near midpoint of straight line
            midpoint = (self.pick_pos + self.place_pos) / 2.0
            jitter = np.random.uniform(-0.02, 0.02, size=3)
            obs_pos = midpoint + jitter
            obs_pos[2] = self.table_z
            self.obstacles.append({
                "color": "red",
                "type": "cup",
                "pos": obs_pos,
                "angle": 0.0
            })

        self.world_objects = [
            {"color": pick_color, "type": pick_type, "pos": self.pick_pos, "angle": pick_angle, "is_wrench": (pick_type == "wrench")},
            {"color": place_color, "type": place_type, "pos": self.place_pos, "angle": 0.0, "is_wrench": False},
        ] + self.obstacles

        # Formulate instruction prompt
        if prompt:
            self.instruction = prompt
        else:
            obs_text = " avoiding the red obstacle" if add_obstacle else ""
            if pick_type == "wrench":
                self.instruction = f"Pick up the {pick_color} wrench and deposit it into the {place_color} {place_type}{obs_text}."
            else:
                self.instruction = f"Transfer the {pick_color} block to the {place_color} {place_type}{obs_text}."

        # Parse instruction via System 2
        self.reasoning_targets = self.reasoning.parse_instruction(self.instruction)

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
        Calculates the exact corrective Delta:
          Delta = (Expert_Joints - Naive_Joints)
        """
        # 1. Naive Path (System 0A Base Solver)
        # Straight line in Cartesian space with yaw=0.0 and no obstacle elevation
        naive_cartesian = []
        naive_joints = []
        
        # Start -> Pick -> Place
        start_pt = self.ee_pos[:3]
        pick_pt = self.pick_pos.copy()
        place_pt = self.place_pos.copy()

        # Build naive path (15 steps to pick, 15 steps to place)
        half_steps = num_steps // 2
        for i in range(half_steps):
            a = i / float(half_steps - 1)
            pt = (1 - a) * start_pt + a * pick_pt
            j = self.kin.inverse(pt[0], pt[1], pt[2], 0.0) # naive yaw = 0.0
            naive_cartesian.append(np.array([pt[0], pt[1], pt[2], 0.0], dtype=np.float32))
            naive_joints.append(j)

        for i in range(half_steps):
            a = i / float(half_steps - 1)
            # Naive goes straight across table plane from pick to place! (Will collide with obstacle)
            pt = (1 - a) * pick_pt + a * place_pt
            j = self.kin.inverse(pt[0], pt[1], pt[2], 0.0)
            naive_cartesian.append(np.array([pt[0], pt[1], pt[2], 0.0], dtype=np.float32))
            naive_joints.append(j)

        naive_cartesian = np.array(naive_cartesian, dtype=np.float32)
        naive_joints = np.array(naive_joints, dtype=np.float32)

        # 2. Expert Path (The ground truth corrective action)
        # Must rotate wrist to match wrench angle, and arc over obstacle in Z
        pick_angle_rad = np.radians(self.world_objects[0]["angle"])
        expert_cartesian = []
        expert_joints = []

        # Reach & Grasp with correct wrist orientation
        for i in range(half_steps):
            a = i / float(half_steps - 1)
            pt = (1 - a) * start_pt + a * pick_pt
            # Correct orientation smoothly aligned to pick_angle_rad
            expert_yaw = a * pick_angle_rad
            j = self.kin.inverse(pt[0], pt[1], pt[2], expert_yaw)
            expert_cartesian.append(np.array([pt[0], pt[1], pt[2], expert_yaw], dtype=np.float32))
            expert_joints.append(j)

        # Transfer with obstacle clearance (arc in Z)
        has_obs = len(self.obstacles) > 0
        for i in range(half_steps):
            a = i / float(half_steps - 1)
            pt = (1 - a) * pick_pt + a * place_pt
            if has_obs:
                # Arc height addition: parabolic elevation to clear cup/obstacle
                arc_z = 0.08 * np.sin(np.pi * a)
                pt[2] += arc_z
            expert_yaw = (1 - a) * pick_angle_rad
            j = self.kin.inverse(pt[0], pt[1], pt[2], expert_yaw)
            expert_cartesian.append(np.array([pt[0], pt[1], pt[2], expert_yaw], dtype=np.float32))
            expert_joints.append(j)

        expert_cartesian = np.array(expert_cartesian, dtype=np.float32)
        expert_joints = np.array(expert_joints, dtype=np.float32)

        # Delta calculation: The corrective delta the neural network must learn to predict!
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

if __name__ == "__main__":
    env = DobotResidualSimEnv()
    for episode in range(3):
        s_img, instr = env.reset()
        demo = env.generate_expert_demonstration(num_steps=10)
        print(f"\nEpisode {episode+1}: {instr}")
        print("Pick Angle:", demo["pick_angle_deg"], "Has Obstacle:", demo["has_obstacle"])
        print("Max Joint Delta (rad):", np.max(np.abs(demo["delta_joints"]), axis=0))
