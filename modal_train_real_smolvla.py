"""
modal_train_real_smolvla.py

Fine-tunes SmolVLA on Real Dobot Demonstrations using Modal Cloud GPU (A10G/T4):
- Uploads real_dobot_demonstrations.pkl directly to the container.
- High-speed GPU training (completes in ~60-90 seconds instead of hours on 4-core CPU).
- Applies continuous SE(2) rotation/shift & photometric augmentations.
- Returns fine-tuned action heads weights directly to disk:
  'smolvla_no_wrist_dobot/smolvla_no_wrist_heads.pt'.
"""

import os
import modal

app = modal.App("train-real-smolvla-dobot")

train_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "torchvision",
        "transformers>=4.48.0",
        "peft>=0.14.0",
        "lerobot",
        "opencv-python-headless",
        "num2words"
    )
    .add_local_file("real_dobot_demonstrations.pkl", "/root/real_dobot_demonstrations.pkl")
)


@app.function(image=train_image, gpu="A10G", timeout=600)
def train_gpu(epochs: int = 10, batch_size: int = 8, lr: float = 3e-4):
    import pickle
    import numpy as np
    import cv2
    import torch
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoTokenizer
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Modal GPU] Initialized on {torch.cuda.get_device_name(0)} ({device})")

    # 1. Load real demonstrations
    with open("/root/real_dobot_demonstrations.pkl", "rb") as f:
        episodes = pickle.load(f)
    print(f"[Modal GPU] Loaded {len(episodes)} real demonstration episodes.")

    transitions = []
    chunk_size = 50

    for ep in episodes:
        actions = np.array(ep["actions"], dtype=np.float32)
        states = np.array(ep["states"], dtype=np.float32)
        cams = ep["cam1"]
        T = len(actions)

        # Multi-Prompt Jitter & Contrastive Synonyms
        instr_base = ep["instructions"][0] if len(ep["instructions"]) > 0 else "approach object"
        prompt_variants = [
            instr_base,
            "approach the lock",
            "move to the object",
            "reach towards the lock",
            "approach target",
            "hover near the lock",
            "go to padlock",
            "navigate to the lock on the desk",
            "move arm towards the metal lock",
            "reach for the target lock",
            "track the lock"
        ]

        for t in range(0, T, 5):  # stride 5 for richer temporal density
            act_chunk = actions[t : t + chunk_size]
            if len(act_chunk) < chunk_size:
                pad_len = chunk_size - len(act_chunk)
                pad_act = np.tile(act_chunk[-1:], (pad_len, 1))
                act_chunk = np.vstack([act_chunk, pad_act])

            transitions.append({
                "cam1_jpg": cams[t],
                "state": states[t],
                "action_chunk": act_chunk,
                "instruction": prompt_variants[t % len(prompt_variants)]
            })

    print(f"[Modal GPU] Generated {len(transitions)} chunked training samples.")

    # 2. PyTorch Dataset with Robust Spatial/Photometric Equivariance
    class RealDobotDataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            item = self.data[idx]
            cam1 = cv2.imdecode(np.frombuffer(item["cam1_jpg"], np.uint8), cv2.IMREAD_COLOR)
            state = item["state"].copy()
            action = item["action_chunk"].copy()
            instr = item["instruction"]

            # Photometric jitter
            if np.random.rand() > 0.3:
                alpha = np.random.uniform(0.8, 1.2)
                beta = np.random.uniform(-20, 20)
                cam1 = np.clip(cam1 * alpha + beta, 0, 255).astype(np.uint8)

            # SE(2) Continuous Rotational & Spatial Equivariance
            if np.random.rand() > 0.2:
                angle_deg = np.random.uniform(-35.0, 35.0)
                angle_rad = np.deg2rad(angle_deg)
                cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
                rot_mat_2d = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)

                action[:, :2] = action[:, :2] @ rot_mat_2d.T

                h, w = cam1.shape[:2]
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

    dataset = RealDobotDataset(transitions)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # 3. Initialize Tokenizer & Model
    print("[Modal GPU] Loading SmolVLM2 tokenizer and base SmolVLA policy...")
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
    policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
    policy.to(device)

    # Freeze vision backbone, unfreeze action heads
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
    print(f"[Modal GPU] Trainable Parameters: {trainable_params / 1e6:.2f}M")

    optimizer = optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)

    # 4. Training Loop
    policy.train()
    print(f"\n[Modal GPU] Starting Training: {epochs} Epochs, {len(loader)} batches/epoch...\n")

    for epoch in range(epochs):
        epoch_loss = 0.0
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
                "observation.language.attention_mask": tokens["attention_mask"].bool(),
                "action": batch["action"].to(device)
            }

            optimizer.zero_grad()
            loss_output = policy(model_batch)
            loss = loss_output[0] if isinstance(loss_output, tuple) else (loss_output["loss"] if isinstance(loss_output, dict) else loss_output)
            loss.backward()

            torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad], max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

            if (step + 1) % 5 == 0 or (step + 1) == len(loader):
                print(f"  Epoch [{epoch+1}/{epochs}] Step [{step+1}/{len(loader)}] | Flow Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / len(loader)
        print(f"--> [Epoch {epoch+1} Complete] Avg Flow Loss: {avg_loss:.4f}\n")

    # 5. Extract trained state dicts (pure CPU dict to send back to local client)
    trained_heads = {
        "state_proj": {k: v.cpu() for k, v in policy.model.state_proj.state_dict().items()},
        "action_in_proj": {k: v.cpu() for k, v in policy.model.action_in_proj.state_dict().items()},
        "action_out_proj": {k: v.cpu() for k, v in policy.model.action_out_proj.state_dict().items()},
        "action_time_mlp_in": {k: v.cpu() for k, v in policy.model.action_time_mlp_in.state_dict().items()},
        "action_time_mlp_out": {k: v.cpu() for k, v in policy.model.action_time_mlp_out.state_dict().items()},
    }
    return trained_heads


@app.local_entrypoint()
def main():
    import torch
    print("=" * 70)
    print("LAUNCHING REAL DOBOT SMOLVLA FINE-TUNING ON MODAL CLOUD GPU (A10G)")
    print("=" * 70)

    heads = train_gpu.remote(epochs=10, batch_size=8, lr=3e-4)

    out_dir = "smolvla_no_wrist_dobot"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "smolvla_no_wrist_heads.pt")
    torch.save(heads, out_file)
    print("\n" + "=" * 70)
    print(f"[SUCCESS] Fine-tuned weights saved locally to: {out_file}")
    print("=" * 70)
