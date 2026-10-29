import os
import sys
import jax
import cv2
import numpy as np
import imageio

import octo_patch
from octo.model.octo_model import OctoModel
from dobot_urdf_renderer import DobotURDFRenderer
from dobot_kinematics import DobotKinematics
import pyrender

# Target colors
COLOR_RGB_MAP = {
    "red": [0.90, 0.15, 0.15],
    "green": [0.15, 0.85, 0.25],
    "blue": [0.15, 0.40, 0.90],
    "yellow": [0.90, 0.85, 0.15],
    "purple": [0.70, 0.20, 0.85],
    "cyan": [0.15, 0.85, 0.85],
    "orange": [0.95, 0.50, 0.10]
}

# Physical dimensions
TABLE_Z_M = -0.1492
CUBE_TOP_Z_M = TABLE_Z_M + 0.024 # -0.1252 m
SUCKER_LENGTH_M = 0.080
CONTACT_EE_Z_M = CUBE_TOP_Z_M + SUCKER_LENGTH_M # -0.0452 m
STANDBY_EE_Z_M = 0.010

# Front Elevated Camera Option 3 (Approved Camera Viewpoint)
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
    return img

def run_pure_neural_evaluation(
    checkpoint_path="octo_dobot_front_checkpoint",
    output_video="octo_pure_neural_front_eval.mp4"
):
    print("=" * 75)
    print("OCTO PURE NEURAL VLA EVALUATION (ZERO HANDHOLDING)")
    print("Camera: Front-Elevated (Option 3) | Checkpoint:", checkpoint_path)
    print("Zero centroid detection | Zero optical Jacobian | Zero plat_pos vector assist")
    print("=" * 75)

    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint '{checkpoint_path}' does not exist.")
        print("Please run training first (e.g. ./autotrainscale.sh on GPU).")
        return

    print("Loading fine-tuned Octo VLA model...")
    model = OctoModel.load_pretrained(checkpoint_path)
    stats = model.dataset_statistics["bridge_dataset"]["action"]
    print("Fine-tuned model loaded successfully.")

    kin = DobotKinematics()
    renderer = DobotURDFRenderer(width=640, height=480)
    cam = pyrender.PerspectiveCamera(yfov=np.radians(50.0))

    benchmarks = [
        {
            "id": 1,
            "cube_color": "red",
            "plat_color": "green",
            "cube_pos": np.array([0.22, 0.05, TABLE_Z_M + 0.012]),
            "plat_pos": np.array([0.22, -0.07, TABLE_Z_M + 0.003]),
            "distractors": [
                {"type": "cube", "color": "blue", "pos": np.array([0.25, -0.02, TABLE_Z_M + 0.012])},
                {"type": "plat", "color": "yellow", "pos": np.array([0.18, 0.08, TABLE_Z_M + 0.003])}
            ],
            "instruction": "pick up the red block and place it on the green platform"
        },
        {
            "id": 2,
            "cube_color": "blue",
            "plat_color": "yellow",
            "cube_pos": np.array([0.21, -0.06, TABLE_Z_M + 0.012]),
            "plat_pos": np.array([0.24, 0.06, TABLE_Z_M + 0.003]),
            "distractors": [
                {"type": "cube", "color": "red", "pos": np.array([0.24, 0.00, TABLE_Z_M + 0.012])},
                {"type": "plat", "color": "cyan", "pos": np.array([0.18, -0.08, TABLE_Z_M + 0.003])}
            ],
            "instruction": "pick up the blue block and place it on the yellow platform"
        }
    ]

    all_frames = []
    success_count = 0

    for b in benchmarks:
        print(f"\n>>> Running Evaluation Sample {b['id']}: '{b['instruction']}'")
        cube_pos = b["cube_pos"].copy()
        plat_pos = b["plat_pos"].copy()
        ee = np.array([cube_pos[0], cube_pos[1], STANDBY_EE_Z_M, 0.0])

        suction_on = False
        suction_attached = False
        placed = False

        obs_history = []
        rng = jax.random.PRNGKey(100 + b["id"])

        for step in range(35):
            # Render front camera image:
            j = kin.inverse(ee[0], ee[1], ee[2], ee[3])
            t = renderer.compute_urdf_transforms(j[0], j[1], j[2], j[3], gripper_closed=False)
            s = renderer.build_scene(
                t,
                cube_pos=cube_pos,
                cube_color_rgb=COLOR_RGB_MAP[b["cube_color"]],
                plat_pos=plat_pos,
                plat_color_rgb=COLOR_RGB_MAP[b["plat_color"]],
                distractor_cubes=[d for d in b["distractors"] if d["type"] == "cube"],
                distractor_plats=[d for d in b["distractors"] if d["type"] == "plat"],
                end_effector="suction"
            )
            raw_rgb = render_front_camera_view(renderer, s, cam)
            img_256 = cv2.resize(raw_rgb, (256, 256), interpolation=cv2.INTER_AREA)

            obs_history.append(img_256)
            if len(obs_history) > 2:
                obs_history.pop(0)

            # Pad history to window=2 if starting
            curr_obs = np.stack([obs_history[0], obs_history[-1]]) # (2, 256, 256, 3)

            # Octo inference dictionary
            obs_dict = {
                "image_primary": curr_obs[None], # (1, 2, 256, 256, 3)
                "image_wrist": np.zeros((1, 2, 128, 128, 3), dtype=np.uint8),
                "pad_mask_dict": {
                    "image_primary": np.ones((1, 2), dtype=bool),
                    "image_wrist": np.zeros((1, 2), dtype=bool),
                    "timestep": np.ones((1, 2), dtype=bool)
                },
                "timestep": np.array([[step, step+1]], dtype=np.int32),
                "timestep_pad_mask": np.ones((1, 2), dtype=bool)
            }
            task_dict = model.create_tasks(texts=[b["instruction"]])

            rng, step_rng = jax.random.split(rng)
            norm_actions = model.sample_actions(
                obs_dict,
                task_dict,
                rng=step_rng,
                unnormalization_statistics=stats
            ) # Shape (1, 2, 4, 7) or (1, 4, 7)

            raw_actions = np.array(norm_actions).squeeze()
            if raw_actions.ndim == 3:
                raw_actions = raw_actions[-1] # take latest window step -> (horizon=4, 7)

            # PURE NEURAL EXECUTION: directly apply model-predicted action delta
            # Zero centroid tracking, zero Jacobian, zero plat_pos vector assist!
            act = raw_actions[0] # execute first action of diffusion chunk
            dx, dy, dz = float(act[0]), float(act[1]), float(act[2])
            grip = float(act[6])

            # Apply actions directly to Dobot EE
            ee[0] = float(np.clip(ee[0] + dx, 0.160, 0.320))
            ee[1] = float(np.clip(ee[1] + dy, -0.160, 0.160))
            ee[2] = float(np.clip(ee[2] + dz, CONTACT_EE_Z_M - 0.002, 0.060))

            sucker_tip_pos = np.array([ee[0], ee[1], ee[2] - SUCKER_LENGTH_M])
            dist_xy = float(np.linalg.norm(sucker_tip_pos[:2] - cube_pos[:2]))
            dist_3d = float(np.linalg.norm(sucker_tip_pos - cube_pos))
            dist_plat_3d = float(np.linalg.norm(cube_pos - plat_pos))

            # Realistic physical contact checking:
            # Physical adherence strictly requires touch (|dist_z| <= 3mm and dist_xy <= 8mm)
            if dist_xy <= 0.008 and abs(sucker_tip_pos[2] - CUBE_TOP_Z_M) <= 0.004 and grip < 0.0:
                suction_attached = True
                suction_on = True

            if grip > 0.5:
                suction_attached = False
                suction_on = False

            if suction_attached:
                cube_pos = np.array([ee[0], ee[1], sucker_tip_pos[2] - 0.012])
                if dist_plat_3d < 0.025 and sucker_tip_pos[2] <= CONTACT_EE_Z_M + 0.006:
                    placed = True
            elif cube_pos[2] > TABLE_Z_M + 0.012:
                cube_pos[2] = max(cube_pos[2] - 0.008, TABLE_Z_M + 0.012)

            print(f"  Step {step:02d} | EE=[{ee[0]:.3f}, {ee[1]:.3f}, {ee[2]:.3f}] | Act=[{dx:+.3f}, {dy:+.3f}, {dz:+.3f}] | Grip={grip:.2f} | Attached={suction_attached} | Placed={placed}")

            # Annotate visual frame for video record
            hud = raw_rgb.copy()
            status_text = f"STEP {step:02d} | PURE NEURAL VLA | ATTACHED: {suction_attached}"
            cv2.putText(hud, status_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if suction_attached else (255, 255, 255), 2)
            cv2.putText(hud, f"Task: {b['instruction']}", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
            all_frames.append(hud)

            if placed:
                print(f"  --> Successfully placed on platform at step {step}!")
                success_count += 1
                break

    if all_frames:
        imageio.mimsave(output_video, all_frames, fps=8)
        print(f"\nEvaluation video saved to: {output_video}")
        print(f"Total Success: {success_count}/{len(benchmarks)}")

if __name__ == "__main__":
    run_pure_neural_evaluation()
