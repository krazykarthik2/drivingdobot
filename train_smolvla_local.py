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

# Fully enforce offline mode so HuggingFace does not attempt network connections
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

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
    chunk_size = 50

    for ep in episodes:
        actions = np.array(ep["actions"], dtype=np.float32)
        states = np.array(ep["states"], dtype=np.float32)
        cams = ep["cam1"]
        T = len(actions)

        instr_base = ep["instructions"][0] if len(ep["instructions"]) > 0 else "approach object"
        prompt_variants = [
            instr_base,
            "approach the lock",
            "move to the object",
            "reach towards the lock",
            "approach target"
        ]

        # Slice into chunks of 50 steps (SmolVLA action horizon chunk_size)
        for t in range(0, T, 10):
            # Extract action sequence [t : t + chunk_size]
            act_chunk = actions[t: t + chunk_size]
            if len(act_chunk) < chunk_size:
                # Pad remaining horizon with the last action
                pad_len = chunk_size - len(act_chunk)
                pad_act = np.tile(act_chunk[-1:], (pad_len, 1))
                act_chunk = np.vstack([act_chunk, pad_act])

            chosen_instr = prompt_variants[t % len(prompt_variants)]
            transitions.append({
                "cam1_jpg": cams[t],
                "state": states[t],
                "action_chunk": act_chunk,
                "instruction": chosen_instr
            })
    print(f"Total chunked transitions: {len(transitions)}")

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
            action = item["action_chunk"].copy()
            instr = item["instruction"]

            # Real-World Generalization & Robust Equivariance Augmentation:
            # 1. Random photometric noise (brightness, contrast, gamma)
            # 2. Random horizontal/lateral shift + rotation to generalize anywhere on desk
            if self.augment:
                # Color jitter
                if np.random.rand() > 0.4:
                    alpha = np.random.uniform(0.8, 1.2) # contrast
                    beta = np.random.uniform(-20, 20)   # brightness
                    cam1 = np.clip(cam1 * alpha + beta, 0, 255).astype(np.uint8)

                # SE(2) Continuous Rotational & Spatial Equivariance
                if np.random.rand() > 0.2:
                    angle_deg = np.random.uniform(-35.0, 35.0)
                    angle_rad = np.deg2rad(angle_deg)
                    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
                    rot_mat_2d = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)

                    action[:, :2] = action[:, :2] @ rot_mat_2d.T

                    h, w = cam1.shape[:2]
                    # Also apply random lateral translation (+/- 25 pixels)
                    dx_t = np.random.uniform(-25.0, 25.0)
                    dy_t = np.random.uniform(-15.0, 15.0)
                    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
                    M[0, 2] += dx_t
                    M[1, 2] += dy_t
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

    print("\nLoading SmolVLM2 tokenizer & SmolVLA policy from local cache...")
    tokenizer_path = r"C:\Users\karthikkrazy\.cache\huggingface\hub\models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct\snapshots\7b375e1b73b11138ff12fe22c8f2822d8fe03467"
    policy_path = r"C:\Users\karthikkrazy\.cache\huggingface\hub\models--lerobot--smolvla_base\snapshots\c83c3163b8ca9b7e67c509fffd9121e66cb96205"

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    policy = SmolVLAPolicy.from_pretrained(policy_path, local_files_only=True)
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

            loss_output = policy(model_batch)
            if isinstance(loss_output, tuple):
                loss = loss_output[0]
            elif isinstance(loss_output, dict):
                loss = loss_output["loss"]
            else:
                loss = loss_output

            loss = loss / grad_accum_steps
            loss.backward()

            epoch_loss += loss.item() * grad_accum_steps

            if (step + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad], max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            print(f"  Epoch [{epoch+1}/{epochs}] Step [{step+1}/{len(loader)}] | Flow Loss: {loss.item() * grad_accum_steps:.4f}", flush=True)

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
