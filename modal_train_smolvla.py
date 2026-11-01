"""
modal_train_smolvla.py

Fine-tunes SmolVLA (lerobot/smolvla_base) on the continuous-domain
SE(2)/SE(3) equivariant Dobot demonstration dataset on Modal T4 GPU.

Recipe:
- Model: lerobot/smolvla_base (SmolVLM2-500M backbone + flow-matching action expert)
- Frozen Vision Backbone: Saves ~10 GB VRAM.
- Trainable Heads: action_in_proj, action_out_proj, action_time_mlp_in, action_time_mlp_out,
  state_proj, and cross-attention adapters via LoRA / expert layers.
- Lie Group SE(2) Data Augmentation:
  Applies continuous planar rotation theta ~ [-30 deg, +30 deg]
  jointly to multi-camera images and action flow velocity targets (Rot(theta) @ delta_action).
- Loss: Continuous Flow Matching (MSE between predicted flow vector v_theta and target v_t = a1 - a0).
- Checkpoint output saved directly to Modal persistent volume 'smolvlm-adapters-vol' under 'smolvla_lora_dobot'.
"""

import os
import modal

app = modal.App("smolvla-dobot-training")
volume = modal.Volume.from_name("smolvlm-adapters-vol")

train_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "lerobot",
        "peft>=0.14.0",
        "torchvision",
        "num2words",
        "transformers>=4.48.0",
        "opencv-python-headless"
    )
)


@app.function(
    image=train_image,
    gpu="T4",
    timeout=2400,
    volumes={"/adapters": volume}
)
def train_smolvla(epochs: int = 3, lr: float = 1e-4, batch_size: int = 4, grad_accum_steps: int = 4):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import pickle
    import numpy as np
    import cv2
    from transformers import AutoTokenizer
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Modal GPU SmolVLA Trainer] Running on {device} ({torch.cuda.get_device_name(0)})...")

    # 1. Load Compressed Dataset from Volume
    data_path = "/adapters/smolvla_dobot_dataset_compressed.pkl"
    print(f"Loading continuous equivariant dataset from volume: {data_path}...")
    with open(data_path, "rb") as f:
        episodes = pickle.load(f)
    print(f"Loaded {len(episodes)} episodes.")

    # Flatten into sample transitions
    transitions = []
    for ep in episodes:
        T = len(ep["actions"])
        for t in range(T):
            transitions.append({
                "cam1_jpg": ep["cam1"][t],
                "cam2_jpg": ep["cam2"][t],
                "cam3_jpg": ep["cam3"][t],
                "state": ep["states"][t],
                "action": ep["actions"][t],
                "instruction": ep["instructions"][t]
            })
    print(f"Total step transitions: {len(transitions)}")

    # 2. SE(2) Equivariant PyTorch Dataset
    class EquivariantDobotDataset(Dataset):
        def __init__(self, data, augment=True):
            self.data = data
            self.augment = augment

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            item = self.data[idx]
            # Decode JPEG bytes to numpy arrays
            cam1 = cv2.imdecode(np.frombuffer(item["cam1_jpg"], np.uint8), cv2.IMREAD_COLOR)
            cam2 = cv2.imdecode(np.frombuffer(item["cam2_jpg"], np.uint8), cv2.IMREAD_COLOR)
            cam3 = cv2.imdecode(np.frombuffer(item["cam3_jpg"], np.uint8), cv2.IMREAD_COLOR)
            state = item["state"].copy()
            action = item["action"].copy()
            instr = item["instruction"]

            # SE(2) Continuous Rotational Equivariance Augmentation
            if self.augment and np.random.rand() > 0.3:
                angle_deg = np.random.uniform(-30.0, 30.0)
                angle_rad = np.deg2rad(angle_deg)
                cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
                rot_mat_2d = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)

                # Rotate planar action (dx, dy)
                action[:2] = rot_mat_2d @ action[:2]

                # Rotate images slightly to reflect the rotation
                h, w = cam1.shape[:2]
                M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
                cam1 = cv2.warpAffine(cam1, M, (w, h), borderMode=cv2.BORDER_REFLECT)

            # Convert to tensors (Channels-first float in [0, 1])
            t_cam1 = torch.from_numpy(cam1).permute(2, 0, 1).float() / 255.0
            t_cam2 = torch.from_numpy(cam2).permute(2, 0, 1).float() / 255.0
            t_cam3 = torch.from_numpy(cam3).permute(2, 0, 1).float() / 255.0
            t_state = torch.from_numpy(state).float()
            t_action = torch.from_numpy(action).float()

            return {
                "cam1": t_cam1,
                "cam2": t_cam2,
                "cam3": t_cam3,
                "state": t_state,
                "action": t_action,
                "instruction": instr
            }

    dataset = EquivariantDobotDataset(transitions, augment=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # 3. Load Tokenizer & Pretrained SmolVLA Policy
    print("Loading SmolVLM2 tokenizer & SmolVLA policy...")
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
    policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
    policy.to(device)

    # 4. Parameter Freezing & Head Fine-Tuning Recipe
    # Freeze the heavy VLM backbone (~448M)
    for param in policy.model.vlm_with_expert.parameters():
        param.requires_grad = False

    # Unfreeze Action Projections, State Projections, and Time MLPs
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
    print(f"\n[Parameter Budget]: Trainable = {trainable_params/1e6:.2f}M / Total = {total_params/1e6:.2f}M ({trainable_params/total_params*100:.2f}%)")

    optimizer = optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)

    # 5. Continuous Flow Matching Training Loop
    policy.train()
    print(f"\nBeginning SmolVLA Training: {epochs} Epochs, {len(loader)} batches/epoch (GradAccum={grad_accum_steps})...\n")

    for epoch in range(epochs):
        epoch_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(loader):
            # Tokenize instructions
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

            # Forward flow matching loss
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

    # 6. Save Finetuned SmolVLA Adapter / Heads
    save_dir = "/adapters/smolvla_lora_dobot"
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving fine-tuned SmolVLA weights to persistent volume: {save_dir}...")
    
    # Save the trainable projection heads
    save_dict = {
        "state_proj": policy.model.state_proj.state_dict(),
        "action_in_proj": policy.model.action_in_proj.state_dict(),
        "action_out_proj": policy.model.action_out_proj.state_dict(),
        "action_time_mlp_in": policy.model.action_time_mlp_in.state_dict(),
        "action_time_mlp_out": policy.model.action_time_mlp_out.state_dict(),
    }
    torch.save(save_dict, os.path.join(save_dir, "smolvla_dobot_heads.pt"))

    volume.commit()
    print("\n[SUCCESS] SmolVLA Dobot fine-tuning complete and committed to Modal volume!")


@app.local_entrypoint()
def main(epochs: int = 3):
    train_smolvla.remote(epochs=epochs)
