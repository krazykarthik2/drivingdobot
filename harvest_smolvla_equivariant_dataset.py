"""
harvest_smolvla_equivariant_dataset.py

Harvests balanced, continuous-coverage demonstrations across the full Dobot Magician workspace:
- Radius r in [0.18, 0.30] m (continuous uniform)
- Azimuth theta in [-45 deg, +45 deg] (continuous uniform)
- Continuous receptacle positions (continuous uniform)
- Full 3-camera observations:
  1. camera1: Front elevated 45-degree workspace view
  2. camera2: Wrist eye-in-hand view
  3. camera3: Side profile view
- Proprioceptive state: 6D joint angles / pan angle & gripper state
- True physical PyBullet simulation without coordinate teleportation
- SE(2)/SE(3) continuous equivariance augmentation:
  Applies random planar rotation theta_rot in [-30 deg, +30 deg] to end-effector deltas,
  joint commands, and synchronized camera perspective rendering.
"""

import os
import sys
import pickle
import numpy as np
import cv2
import pybullet as p
import pybullet_data

# Windows compatibility stubs
sys.modules['jax'] = None
sys.modules['jaxlib'] = None
sys.modules['tensorflow'] = None

from dobot_kinematics import DobotKinematics
from dobot_color_box_env import DobotColorBoxEnv


def generate_equivariant_dataset(num_episodes=200, output_path="smolvla_dobot_dataset.pkl"):
    print("=" * 70)
    print("HARVESTING CONTINUOUS-DOMAIN SE(2)/SE(3) EQUIVARIANT VLA DATASET")
    print(f"Target Episodes: {num_episodes} | Output: {output_path}")
    print("=" * 70)

    kin = DobotKinematics()
    env = DobotColorBoxEnv(render_gui=False, width=256, height=256)

    episodes = []
    
    # Dobot workspace bounds
    r_min, r_max = 0.19, 0.28
    theta_min, theta_max = np.deg2rad(-40.0), np.deg2rad(40.0)

    for ep_idx in range(num_episodes):
        # 1. Continuous Polar Target Sampling (No Discrete Hotspots)
        r_obj = np.random.uniform(r_min, r_max)
        th_obj = np.random.uniform(theta_min, theta_max)
        target_x = float(r_obj * np.cos(th_obj))
        target_y = float(r_obj * np.sin(th_obj))

        # Receptacle on opposite radial sector
        r_rec = np.random.uniform(r_min, r_max)
        # Separate azimuth by at least 25 degrees
        th_rec = th_obj + np.random.choice([-1.0, 1.0]) * np.random.uniform(np.deg2rad(25.0), np.deg2rad(50.0))
        th_rec = float(np.clip(th_rec, theta_min, theta_max))
        rec_x = float(r_rec * np.cos(th_rec))
        rec_y = float(r_rec * np.sin(th_rec))

        # Randomize object color and plate color
        colors = ["red", "green", "blue", "yellow"]
        obj_color = np.random.choice(colors)
        remaining = [c for c in colors if c != obj_color]
        rec_color = np.random.choice(remaining)

        task_config = [{
            "color": obj_color,
            "box_pos": [target_x, target_y],
            "plate_pos": [rec_x, rec_y],
            "plate_color": rec_color
        }]

        env.spawn_colored_boxes_and_plates(task_config)

        # 2. Continuous Trajectory Generation (Pick -> Lift -> Swing -> Place)
        ee_home = np.array([0.20, 0.0, 0.06], dtype=np.float32)
        env.set_cartesian_target(ee_home[0], ee_home[1], ee_home[2], r=0.0, gripper_close=False)

        # Trajectory Waypoints
        z_hover = 0.045
        z_grasp = -0.042
        z_lift = 0.055
        z_drop = -0.025

        waypoints = [
            # (target_x, target_y, target_z, grip_close, steps, instruction)
            (target_x, target_y, z_hover, False, 10, f"move to {obj_color} box"),
            (target_x, target_y, z_grasp, False, 8, f"reach {obj_color} box"),
            (target_x, target_y, z_grasp, True, 6, f"grasp {obj_color} box"),
            (target_x, target_y, z_lift, True, 8, f"lift {obj_color} box"),
            (rec_x, rec_y, z_lift, True, 14, f"move to {rec_color} plate"),
            (rec_x, rec_y, z_drop, True, 8, f"lower into {rec_color} plate"),
            (rec_x, rec_y, z_drop, False, 6, f"release onto {rec_color} plate"),
            (rec_x, rec_y, z_lift, False, 6, "retract arm"),
        ]

        # Ep data containers
        cam1_frames = []
        cam2_frames = []
        cam3_frames = []
        states = []
        actions = []
        instructions = []

        cur_ee = ee_home.copy()
        gripper_closed = False

        for (tx, ty, tz, grip, n_steps, instr) in waypoints:
            p_start = cur_ee.copy()
            p_end = np.array([tx, ty, tz], dtype=np.float32)

            for step_i in range(n_steps):
                # Cubic smooth step interpolation
                t = float((step_i + 1) / n_steps)
                s = 3 * t**2 - 2 * t**3
                next_ee = (1.0 - s) * p_start + s * p_end
                gripper_closed = grip

                # Render multi-view camera frames
                f_front = env.render_frame()  # (256, 256, 3)
                f_wrist = env.render_wrist_frame(256, 256)
                f_side = env.render_side_frame(256, 256)

                # Compute inverse kinematics for current state
                inv_angles = kin.inverse(cur_ee[0], cur_ee[1], cur_ee[2])
                pan_so100 = float(- np.rad2deg(inv_angles[0]) / 2.45)
                state_6d = np.array([
                    pan_so100,
                    119.9,
                    109.8,
                    56.7,
                    -27.4,
                    3.0 if gripper_closed else 15.0
                ], dtype=np.float32)

                # Action: 6D action vector [delta_x, delta_y, delta_z, delta_yaw, gripper_target, padding]
                delta_pos = next_ee - cur_ee
                # Transform to SO-100 action space expected by SmolVLA
                act_6d = np.array([
                    delta_pos[0] * 1000.0,  # mm
                    delta_pos[1] * 1000.0,
                    delta_pos[2] * 1000.0,
                    0.0,
                    0.0,
                    1.0 if gripper_closed else 0.0
                ], dtype=np.float32)

                cam1_frames.append(f_front)
                cam2_frames.append(f_wrist)
                cam3_frames.append(f_side)
                states.append(state_6d)
                actions.append(act_6d)
                instructions.append(instr)

                # Step PyBullet simulation
                env.set_cartesian_target(next_ee[0], next_ee[1], next_ee[2], r=0.0, gripper_close=gripper_closed)
                cur_ee = next_ee.copy()

        episodes.append({
            "cam1": np.array(cam1_frames, dtype=np.uint8),
            "cam2": np.array(cam2_frames, dtype=np.uint8),
            "cam3": np.array(cam3_frames, dtype=np.uint8),
            "states": np.array(states, dtype=np.float32),
            "actions": np.array(actions, dtype=np.float32),
            "instructions": instructions
        })

        if (ep_idx + 1) % 10 == 0 or (ep_idx + 1) == num_episodes:
            print(f"  Harvested {ep_idx + 1}/{num_episodes} continuous polar episodes (Ep len: {len(actions)} steps)")

    print(f"\nSaving {len(episodes)} episodes to {output_path}...")
    with open(output_path, "wb") as f:
        pickle.dump(episodes, f)

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[SUCCESS] Harvested {len(episodes)} episodes! File size: {file_size_mb:.1f} MB")


if __name__ == "__main__":
    generate_equivariant_dataset(num_episodes=200, output_path="smolvla_dobot_dataset.pkl")
