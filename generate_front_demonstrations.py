import os
import sys
import pickle
import numpy as np
import cv2
import pyrender
import octo_patch
from dobot_urdf_renderer import DobotURDFRenderer
from dobot_kinematics import DobotKinematics

# Target colors
COLOR_RGB_MAP = {
    "red": [0.90, 0.15, 0.15],
    "green": [0.15, 0.85, 0.25],
    "blue": [0.15, 0.40, 0.90],
    "yellow": [0.90, 0.85, 0.15],
    "cyan": [0.15, 0.85, 0.85],
    "purple": [0.70, 0.20, 0.85],
    "orange": [0.95, 0.50, 0.10]
}

TABLE_Z_M = -0.1492
CUBE_TOP_Z_M = TABLE_Z_M + 0.024 # -0.1252 m
SUCKER_LENGTH_M = 0.080
CONTACT_EE_Z_M = CUBE_TOP_Z_M + SUCKER_LENGTH_M # -0.0452 m
SAFE_LIFT_Z_M = 0.020 # Safe transport hover height

# Front Elevated Camera Option 3
CAM_EYE = np.array([0.44, 0.00, 0.28])
CAM_TGT = np.array([0.22, 0.00, -0.10])
CAM_UP = np.array([0, 0, 1])
fwd = (CAM_TGT - CAM_EYE) / np.linalg.norm(CAM_TGT - CAM_EYE)
rgt = np.cross(fwd, CAM_UP) / np.linalg.norm(np.cross(fwd, CAM_UP))
cam_up = np.cross(rgt, fwd)

T_CAM = np.eye(4)
T_CAM[:3, :3] = np.column_stack([rgt, cam_up, -fwd])
T_CAM[:3, 3] = CAM_EYE

def render_front_camera_view(renderer, scene, cam):
    cam_node = scene.add(cam, pose=T_CAM)
    light_node = scene.add(pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.5), pose=T_CAM)
    img, _ = renderer.r.render(scene)
    scene.remove_node(cam_node)
    scene.remove_node(light_node)
    return cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)

def generate_expert_demonstrations(num_episodes=120, output_path="dobot_front_demonstrations.pkl"):
    print("=" * 65)
    print("GENERATING EXPERT DEMONSTRATIONS FROM FRONT ELEVATED CAMERA")
    print(f"Target Episodes: {num_episodes} | Output: {output_path}")
    print("=" * 65)

    kin = DobotKinematics()
    renderer = DobotURDFRenderer(width=640, height=480)
    cam = pyrender.PerspectiveCamera(yfov=np.radians(50.0))

    tasks = [
        {"cube": "red", "plat": "green", "text": "pick up the red block and place it on the green platform"},
        {"cube": "blue", "plat": "yellow", "text": "pick up the blue block and place it on the yellow platform"},
        {"cube": "yellow", "plat": "cyan", "text": "pick up the yellow block and place it on the cyan platform"},
        {"cube": "green", "plat": "red", "text": "pick up the green block and place it on the red platform"}
    ]

    demonstrations = []
    np.random.seed(42)

    for ep_idx in range(num_episodes):
        task_info = tasks[ep_idx % len(tasks)]
        c_name = task_info["cube"]
        p_name = task_info["plat"]
        instruction = task_info["text"]

        # Randomize placement on table:
        # Cube Y in [-0.08, +0.08], X in [0.19, 0.25]
        cube_x = np.random.uniform(0.19, 0.25)
        cube_y = np.random.uniform(-0.08, 0.08)
        cube_pos = np.array([cube_x, cube_y, TABLE_Z_M + 0.012])

        # Platform opposite side or random distant spot
        plat_x = np.random.uniform(0.19, 0.25)
        plat_y = -np.sign(cube_y) * np.random.uniform(0.05, 0.09) if abs(cube_y) > 0.02 else np.random.choice([-0.07, 0.07])
        plat_pos = np.array([plat_x, plat_y, TABLE_Z_M + 0.003])

        # Initial EE starts above cube:
        ee = np.array([cube_x, cube_y, 0.010, 0.0])

        episode_obs = []
        episode_acts = []
        suction_attached = False

        # Phase 1: Descent to cube contact (7 steps)
        # Phase 2: Lift block to safe transport clearance (4 steps)
        # Phase 3: Transport across table to platform (8-12 steps)
        # Phase 4: Descend onto platform and release (5 steps)

        waypoints = []
        # Waypoints for EE:
        # 1. Approach descent:
        for z in np.linspace(0.010, CONTACT_EE_Z_M, 6):
            waypoints.append((np.array([cube_x, cube_y, z]), 0.0)) # grip=0.0 (suction off)
        # Contact step:
        waypoints.append((np.array([cube_x, cube_y, CONTACT_EE_Z_M]), -1.0)) # grip=-1.0 (suction ON)

        # 2. Lift:
        for z in np.linspace(CONTACT_EE_Z_M, SAFE_LIFT_Z_M, 4):
            waypoints.append((np.array([cube_x, cube_y, z]), -1.0))

        # 3. Transport to platform:
        n_transport = 10
        xs = np.linspace(cube_x, plat_x, n_transport)
        ys = np.linspace(cube_y, plat_y, n_transport)
        for i in range(n_transport):
            waypoints.append((np.array([xs[i], ys[i], SAFE_LIFT_Z_M]), -1.0))

        # 4. Descent & release on platform:
        for z in np.linspace(SAFE_LIFT_Z_M, CONTACT_EE_Z_M, 5):
            waypoints.append((np.array([plat_x, plat_y, z]), -1.0))
        # Release step:
        waypoints.append((np.array([plat_x, plat_y, CONTACT_EE_Z_M]), 1.0)) # grip=1.0 (release)

        # Execute trajectory and record front camera frames:
        current_ee = ee.copy()
        current_cube = cube_pos.copy()

        for w_idx in range(len(waypoints) - 1):
            next_pos, grip_val = waypoints[w_idx + 1]
            # Compute action delta:
            d_pos = next_pos - current_ee[:3]
            act_7d = np.array([d_pos[0], d_pos[1], d_pos[2], 0.0, 0.0, 0.0, grip_val], dtype=np.float32)

            # Render scene from front camera:
            j = kin.inverse(current_ee[0], current_ee[1], current_ee[2], current_ee[3])
            t = renderer.compute_urdf_transforms(j[0], j[1], j[2], j[3], gripper_closed=False)
            s = renderer.build_scene(
                t,
                cube_pos=current_cube,
                cube_color_rgb=COLOR_RGB_MAP[c_name],
                plat_pos=plat_pos,
                plat_color_rgb=COLOR_RGB_MAP[p_name],
                distractor_cubes=[],
                distractor_plats=[],
                end_effector="suction"
            )
            img_front = render_front_camera_view(renderer, s, cam)

            episode_obs.append(img_front)
            episode_acts.append(act_7d)

            # Update dynamics:
            current_ee[:3] = next_pos
            if grip_val < -0.5:
                suction_attached = True
            elif grip_val > 0.5:
                suction_attached = False

            if suction_attached:
                current_cube = np.array([current_ee[0], current_ee[1], current_ee[2] - SUCKER_LENGTH_M + 0.012])

        demonstrations.append({
            "instruction": instruction,
            "observations": np.stack(episode_obs), # (T, 256, 256, 3)
            "actions": np.stack(episode_acts)      # (T, 7)
        })

        if (ep_idx + 1) % 20 == 0:
            print(f"  Generated {ep_idx + 1}/{num_episodes} demonstration episodes...")

    with open(output_path, "wb") as f:
        pickle.dump(demonstrations, f)
    print(f"Successfully generated and saved {len(demonstrations)} episodes to {output_path}!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate Dobot Front Camera Demonstrations")
    parser.add_argument("--episodes", type=int, default=80, help="Number of demonstration episodes to generate (e.g. 1000 or 10000)")
    parser.add_argument("--output", type=str, default="dobot_front_demonstrations.pkl", help="Output pickle path")
    args = parser.parse_args()

    generate_expert_demonstrations(num_episodes=args.episodes, output_path=args.output)
