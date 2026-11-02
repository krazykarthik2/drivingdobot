"""
train_smolvla_local.py

Trains SmolVLA without wrist camera locally using PyTorch.
- Loads smolvla_dobot_dataset_compressed.pkl (246 MB, already on local disk).
- Freezes heavy vision backbone (~448M).
- Trains action & projection heads (~12M params).
- Runs directly on local hardware without any network/cloud connection drops.
"""

import os
import sys
import pickle
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Windows compatibility stubs
sys.modules['jax'] = None
sys.modules['jaxlib'] = None
sys.modules['tensorflow'] = None

import huggingface_hub
if not hasattr(huggingface_hub, 'sync_bucket'):
    huggingface_hub.sync_bucket = None

import transformers.utils
if not hasattr(transformers.utils, 'torch_compilable_check'):
    transformers.utils.torch_compilable_check = lambda: False

from transformers import AutoTokenizer
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


def train_local(data_path="real_dobot_demonstrations.pkl", epochs=3, batch_size=4, grad_accum_steps=2, lr=2e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 65)
    print(f"LOCAL SMOLVLA FINE-TUNING (NO WRIST CAMERA)")
    print(f"Device: {device} | Dataset: {data_path}")
    print("=" * 65)

    if not os.path.exists(data_path):
        if os.path.exists("smolvla_dobot_dataset_compressed.pkl"):
            data_path = "smolvla_dobot_dataset_compressed.pkl"
        else:
            raise FileNotFoundError(f"{data_path} not found!")

    print(f"Loading dataset from {data_path}...")
    with open(data_path, "rb") as f:
        episodes = pickle.load(f)
    print(f"Loaded {len(episodes)} episodes.")

    transitions = []
    for ep in episodes:
        T = len(ep["actions"])
        for t in range(T):
            transitions.append({
                "cam1_jpg": ep["cam1"][t],
                "state": ep["states"][t],
                "action": ep["actions"][t],
                "instruction": ep["instructions"][t]
            })
    print(f"Total transitions: {len(transitions)}")

    # SE(2) Equivariant PyTorch Dataset (Zero Wrist Camera)
    class LocalNoWristDataset(Dataset):
        def __init__(self, data, augment=True):
            self.data = data
            self.augment = augment

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            item = self.data[idx]
            cam1 = cv2.imdecode(np.frombuffer(item["cam1_jpg"], np.uint8), cv2.IMREAD_COLOR)
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

    dataset = LocalNoWristDataset(transitions, augment=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    print("\nLoading SmolVLM2 tokenizer & SmolVLA policy...")
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
    policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
    policy.to(device)

    # Freeze vision backbone, train action flow heads
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
    print(f"[Parameter Budget]: Trainable = {trainable_params/1e6:.2f}M / Total = {total_params/1e6:.2f}M")

    optimizer = optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)

    policy.train()
    print(f"\nBeginning Training: {epochs} Epochs, {len(loader)} batches/epoch (GradAccum={grad_accum_steps})...\n")

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

            if (step + 1) % 50 == 0 or (step + 1) == len(loader):
                print(f"  Epoch [{epoch+1}/{epochs}] Step [{step+1}/{len(loader)}] | Flow Loss: {loss.item() * grad_accum_steps:.4f}")

        avg_loss = epoch_loss / len(loader)
        print(f"[Epoch {epoch+1} Complete] Average Flow Loss: {avg_loss:.4f}\n")

    out_dir = "smolvla_no_wrist_dobot"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "smolvla_no_wrist_heads.pt")
    print(f"Saving fine-tuned heads to {out_file}...")
    save_dict = {
        "state_proj": policy.model.state_proj.state_dict(),
        "action_in_proj": policy.model.action_in_proj.state_dict(),
        "action_out_proj": policy.model.action_out_proj.state_dict(),
        "action_time_mlp_in": policy.model.action_time_mlp_in.state_dict(),
        "action_time_mlp_out": policy.model.action_time_mlp_out.state_dict(),
    }
    torch.save(save_dict, out_file)
    print("\n[SUCCESS] Local fine-tuning without wrist camera complete!")


if __name__ == "__main__":
    train_local()
