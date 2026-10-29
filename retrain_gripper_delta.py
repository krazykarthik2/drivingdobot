"""
Retrains the Neural Delta Driver specifically calibrated for Gripper Physics.
Ensures strong corrective delta for:
1. Object orientation matching (wrist yaw alignment).
2. True Obstacle clearance (+0.100m parabolic arc).
"""

import os
import sys
import io
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from dobot_residual_env import DobotResidualSimEnv
from neural_delta_model import NeuralDeltaDriver

class ResidualDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        img = item["silhouette"].transpose(2, 0, 1).astype(np.float32) / 255.0
        naive_j = item["naive_j"].astype(np.float32)
        target_delta = item["delta_j"].astype(np.float32)
        return torch.tensor(img), torch.tensor(naive_j), torch.tensor(target_delta)

def collect_gripper_dataset(num_episodes=250, steps_per_ep=30):
    print(f"Collecting Gripper Shadow Dataset: {num_episodes} episodes x {steps_per_ep} waypoints...")
    env = DobotResidualSimEnv(table_z=0.070, standby_z=0.150)
    dataset_records = []
    
    for ep in range(num_episodes):
        has_obs = (ep % 2 == 0)
        env.reset(add_obstacle=has_obs, is_weird_shape=True)
        demo = env.generate_expert_demonstration(num_steps=steps_per_ep)
        
        sil = demo["silhouette_128"]
        for t in range(steps_per_ep):
            dataset_records.append({
                "silhouette": sil,
                "naive_j": demo["naive_joints"][t],
                "delta_j": demo["delta_joints"][t]
            })

    print(f"Total shadow samples collected: {len(dataset_records)}")
    return dataset_records

def train_gripper_delta_network(num_epochs=15, batch_size=32, lr=1e-3):
    data = collect_gripper_dataset(num_episodes=250, steps_per_ep=30)
    dataset = ResidualDataset(data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Gripper Neural Delta Driver on device: {device}")
    
    model = NeuralDeltaDriver().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    model.train()
    start_time = time.time()
    for epoch in range(1, num_epochs + 1):
        total_loss = 0.0
        for imgs, naive_j, target_delta in loader:
            imgs = imgs.to(device)
            naive_j = naive_j.to(device)
            target_delta = target_delta.to(device)
            
            optimizer.zero_grad()
            pred_delta = model(imgs, naive_j)
            loss = criterion(pred_delta, target_delta)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(imgs)
            
        epoch_loss = total_loss / len(dataset)
        print(f"Epoch [{epoch:02d}/{num_epochs:02d}] - Gripper Residual MSE Loss: {epoch_loss:.6f}")

    print(f"Training finished in {time.time() - start_time:.2f}s!")
    torch.save(model.state_dict(), "dobot_delta_model.pt")
    print("Saved updated weights to dobot_delta_model.pt")

if __name__ == "__main__":
    train_gripper_delta_network()
