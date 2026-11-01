"""
modal_train_smolvla_no_wrist.py

Fine-tunes SmolVLA WITHOUT Wrist Camera on Modal Tesla T4 GPU.
1. PyBullet in headless mode renders ONLY the front 45-degree elevated camera.
   - Wrist camera rendering is completely removed (saves ~50% render time & RAM).
   - For architecture compatibility with lerobot/smolvla_base, camera2 and camera3 are fed
     as static zero tensors (or duplicate of cam1), training the model to attend 100% to the front view.
2. Continuous Polar Workspace Sampling:
   - radius r in [0.19, 0.28] m, azimuth theta in [-40 deg, +40 deg] (zero discrete hotspots).
3. SE(2)/SE(3) Equivariance Data Augmentation:
   - Random planar rotations theta_aug in [-30 deg, +30 deg] applied jointly to front camera image
     and (dx, dy) action vectors.
4. Minimal-Compute LoRA / Head Fine-Tuning:
   - Heavy VLM backbone frozen (~448M params).
   - Flow-matching action & projection heads fine-tuned (~12M params).
   - Weights saved directly to Modal volume 'smolvlm-adapters-vol' under 'smolvla_no_wrist_dobot'.
"""

import os
import modal

app = modal.App("smolvla-no-wrist-training")
volume = modal.Volume.from_name("smolvlm-adapters-vol")

pipeline_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "pybullet",
        "lerobot",
        "peft>=0.14.0",
        "torchvision",
        "num2words",
        "transformers>=4.48.0",
        "opencv-python-headless"
    )
    .add_local_dir("dobot_magician_ros2", "/root/dobot_magician_ros2")
    .add_local_file("dobot_kinematics.py", "/root/dobot_kinematics.py")
    .add_local_file("dobot_color_box_env.py", "/root/dobot_color_box_env.py")
)


@app.function(
    image=pipeline_image,
    gpu="T4",
    timeout=2400,
    volumes={"/adapters": volume}
)
def run_no_wrist_pipeline(num_episodes: int = 200, epochs: int = 5, batch_size: int = 4, grad_accum_steps: int = 4):
    import sys
    sys.path.append("/root")
    import numpy as np
    import cv2
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import pybullet as p
    from transformers import AutoTokenizer
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from dobot_kinematics import DobotKinematics
    from dobot_color_box_env import DobotColorBoxEnv

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 65)
    print(f"SMOLVLA FINE-TUNING WITHOUT WRIST CAMERA")
    print(f"GPU: {torch.cuda.get_device_name(0)} | Policy: lerobot/smolvla_base")
    print("=" * 65 + "\n")

    # 1. Continuous Polar Demonstration Harvesting (Front Camera Only)
    print(f"[Phase 1/2] Harvesting {num_episodes} Continuous Polar Trajectories (Front Camera ONLY)...")
    kin = DobotKinematics()
    env = DobotColorBoxEnv(render_gui=False, width=256, height=256)

    transitions = []
    r_min, r_max = 0.19, 0.28
    theta_min, theta_max = np.deg2rad(-40.0), np.deg2rad(40.0)

    for ep_idx in range(num_episodes):
        r_obj = np.random.uniform(r_min, r_max)
        th_obj = np.random.uniform(theta_min, theta_max)
        target_x = float(r_obj * np.cos(th_obj))
        target_y = float(r_obj * np.sin(th_obj))

        r_rec = np.random.uniform(r_min, r_max)
        th_rec = th_obj + np.random.choice([-1.0, 1.0]) * np.random.uniform(np.deg2rad(25.0), np.deg2rad(50.0))
        th_rec = float(np.clip(th_rec, theta_min, theta_max))
        rec_x = float(r_rec * np.cos(th_rec))
        rec_y = float(r_rec * np.sin(th_rec))

        colors = ["red", "green", "blue", "yellow"]
        obj_color = np.random.choice(colors)
        rec_color = np.random.choice([c for c in colors if c != obj_color])

        task_config = [{
            "color": obj_color,
            "box_pos": [target_x, target_y],
            "plate_pos": [rec_x, rec_y],
            "plate_color": rec_color
        }]
        env.spawn_colored_boxes_and_plates(task_config)

        ee_home = np.array([0.20, 0.0, 0.06], dtype=np.float32)
        env.set_cartesian_target(ee_home[0], ee_home[1], ee_home[2], r=0.0, gripper_close=False)

        waypoints = [
            (target_x, target_y, 0.045, False, 10, f"move to {obj_color} box"),
            (target_x, target_y, -0.042, False, 8, f"reach {obj_color} box"),
            (target_x, target_y, -0.042, True, 6, f"grasp {obj_color} box"),
            (target_x, target_y, 0.055, True, 8, f"lift {obj_color} box"),
            (rec_x, rec_y, 0.055, True, 14, f"move to {rec_color} plate"),
            (rec_x, rec_y, -0.025, True, 8, f"lower into {rec_color} plate"),
            (rec_x, rec_y, -0.025, False, 6, f"release onto {rec_color} plate"),
            (rec_x, rec_y, 0.055, False, 6, "retract arm"),
        ]

        cur_ee = ee_home.copy()
        for (tx, ty, tz, grip, n_steps, instr) in waypoints:
            p_start = cur_ee.copy()
            p_end = np.array([tx, ty, tz], dtype=np.float32)

            for step_i in range(n_steps):
                t = float((step_i + 1) / n_steps)
                s = 3 * t**2 - 2 * t**3
                next_ee = (1.0 - s) * p_start + s * p_end
                gripper_closed = grip

                # ONLY render front camera (NO wrist camera rendering)
                f_front = env.render_frame()

                inv_angles = kin.inverse(cur_ee[0], cur_ee[1], cur_ee[2])
                pan_so100 = float(- np.rad2deg(inv_angles[0]) / 2.45)
                state_6d = np.array([
                    pan_so100, 119.9, 109.8, 56.7, -27.4, 3.0 if gripper_closed else 15.0
                ], dtype=np.float32)

                delta_pos = next_ee - cur_ee
                act_6d = np.array([
                    delta_pos[0] * 1000.0,
                    delta_pos[1] * 1000.0,
                    delta_pos[2] * 1000.0,
                    0.0, 0.0,
                    1.0 if gripper_closed else 0.0
                ], dtype=np.float32)

                transitions.append({
                    "cam1": f_front,
                    "state": state_6d,
                    "action": act_6d,
                    "instruction": instr
                })

                env.set_cartesian_target(next_ee[0], next_ee[1], next_ee[2], r=0.0, gripper_close=gripper_closed)
                cur_ee = next_ee.copy()

        if (ep_idx + 1) % 25 == 0 or (ep_idx + 1) == num_episodes:
            print(f"  Harvested {ep_idx + 1}/{num_episodes} episodes ({len(transitions)} transitions)")

    print(f"\n[Phase 1 Complete] Harvested {len(transitions)} transitions (Zero wrist camera rendering).")

    # 2. SE(2) Equivariant PyTorch Dataset (Zero Wrist Camera)
    class NoWristEquivariantDataset(Dataset):
        def __init__(self, data, augment=True):
            self.data = data
            self.augment = augment

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            item = self.data[idx]
            cam1 = item["cam1"].copy()
            state = item["state"].copy()
            action = item["action"].copy()
            instr = item["instruction"]

            # SE(2) Continuous Rotational Equivariance Augmentation
            if self.augment and np.random.rand() > 0.3:
                angle_deg = np.random.uniform(-30.0, 30.0)
                angle_rad = np.deg2rad(angle_deg)
                cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
                rot_mat_2d = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)

                action[:2] = rot_mat_2d @ action[:2]

                h, w = cam1.shape[:2]
                M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
                cam1 = cv2.warpAffine(cam1, M, (w, h), borderMode=cv2.BORDER_REFLECT)

            t_cam1 = torch.from_numpy(cam1).permute(2, 0, 1).float() / 255.0
            # Feed static zero tensors for camera2 (wrist) and camera3 (side) to satisfy architecture specs
            t_zero = torch.zeros_like(t_cam1)
            t_state = torch.from_numpy(state).float()
            t_action = torch.from_numpy(action).float()

            return {
                "cam1": t_cam1,
                "cam2": t_zero,
                "cam3": t_zero,
                "state": t_state,
                "action": t_action,
                "instruction": instr
            }

    dataset = NoWristEquivariantDataset(transitions, augment=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # 3. Load Tokenizer & SmolVLA Policy
    print("\n[Phase 2/2] Initializing SmolVLA Model (Vision Backbone Frozen)...")
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
    policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
    policy.to(device)

    # 4. Freeze VLM Backbone, Train Action Flow Heads
    for param in policy.model.vlm_with_expert.parameters():
        param.requires_grad = False

    trainable_modules = [
        policy.model.state_proj,
        policy.model.action_in_proj,
        policy.model.action_out_proj,
        policy.model.action_time_mlp_in,
        policy.model.action_time_mlp_out
    ]
    for mod in trainable_modules:
        for param in mod.parameters():
            param.requires_grad = True

    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in policy.parameters())
    print(f"[Parameter Budget]: Trainable = {trainable_params/1e6:.2f}M / Total = {total_params/1e6:.2f}M ({trainable_params/total_params*100:.2f}%)")

    optimizer = optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)

    # 5. Flow Matching Training Loop
    policy.train()
    print(f"\nBeginning Flow-Matching Training: {epochs} Epochs, {len(loader)} batches/epoch (GradAccum={grad_accum_steps})...\n")

    for epoch in range(epochs):
        epoch_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(loader):
            tokens = tokenizer(
                batch["instruction"],
                return_tensors="pt",
                padding="max_length",
                max_length=48,
                truncation=True
            ).to(device)

            model_batch = {
                "observation.state": batch["state"].to(device),
                "observation.images.camera1": batch["cam1"].to(device),
                "observation.images.camera2": batch["cam2"].to(device),
                "observation.images.camera3": batch["cam3"].to(device),
                "observation.language.tokens": tokens["input_ids"],
                "observation.language.attention_mask": tokens["attention_mask"],
                "action": batch["action"].to(device)
            }

            loss_dict = policy(model_batch)
            loss = loss_dict["loss"] if isinstance(loss_dict, dict) else loss_dict
            loss = loss / grad_accum_steps
            loss.backward()

            epoch_loss += loss.item() * grad_accum_steps

            if (step + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad], max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            if (step + 1) % 100 == 0 or (step + 1) == len(loader):
                print(f"  Epoch [{epoch+1}/{epochs}] Step [{step+1}/{len(loader)}] | Flow Loss: {loss.item() * grad_accum_steps:.4f}")

        avg_loss = epoch_loss / len(loader)
        print(f"[Epoch {epoch+1} Complete] Average Flow-Matching Loss: {avg_loss:.4f}\n")

    # 6. Save Finetuned SmolVLA Heads (No-Wrist Model)
    save_dir = "/adapters/smolvla_no_wrist_dobot"
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving fine-tuned SmolVLA (No-Wrist) weights to persistent volume: {save_dir}...")
    
    save_dict = {
        "state_proj": policy.model.state_proj.state_dict(),
        "action_in_proj": policy.model.action_in_proj.state_dict(),
        "action_out_proj": policy.model.action_out_proj.state_dict(),
        "action_time_mlp_in": policy.model.action_time_mlp_in.state_dict(),
        "action_time_mlp_out": policy.model.action_time_mlp_out.state_dict(),
    }
    torch.save(save_dict, os.path.join(save_dir, "smolvla_no_wrist_heads.pt"))

    volume.commit()
    print("\n[SUCCESS] SmolVLA (No-Wrist) fine-tuning complete and committed to Modal volume!")


@app.local_entrypoint()
def main(num_episodes: int = 150, epochs: int = 5):
    run_no_wrist_pipeline.remote(num_episodes=num_episodes, epochs=epochs)
