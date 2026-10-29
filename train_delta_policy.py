import os
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
        # silhouette image normalized to [0, 1] float32: [3, 128, 128]
        img = item["silhouette"].transpose(2, 0, 1).astype(np.float32) / 255.0
        naive_j = item["naive_j"].astype(np.float32)
        target_delta = item["delta_j"].astype(np.float32)
        return torch.tensor(img), torch.tensor(naive_j), torch.tensor(target_delta)

def collect_shadow_dataset(num_episodes=150, steps_per_ep=20):
    print(f"Collecting Phase 2 Shadow Data: {num_episodes} episodes x {steps_per_ep} waypoints...")
    env = DobotResidualSimEnv()
    dataset_records = []
    
    for ep in range(num_episodes):
        # Alternate with and without obstacle, with and without wrench
        has_obs = (ep % 2 == 0)
        is_wrench = True
        env.reset(add_obstacle=has_obs, is_weird_shape=is_wrench)
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

def train_delta_network(num_epochs=12, batch_size=32, lr=1e-3, onnx_export_path="dobot_delta_policy.onnx"):
    data = collect_shadow_dataset(num_episodes=200, steps_per_ep=20)
    dataset = ResidualDataset(data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Neural Delta Driver on device: {device}")
    
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
        print(f"Epoch [{epoch:02d}/{num_epochs:02d}] - Residual MSE Loss: {epoch_loss:.6f}")

    elapsed = time.time() - start_time
    print(f"Training complete in {elapsed:.2f} seconds!")
    
    # Save PyTorch weights
    torch.save(model.state_dict(), "dobot_delta_model.pt")
    print("Saved PyTorch model weights to dobot_delta_model.pt")

    # Export to ONNX for Jetson Nano edge deployment
    model.eval()
    dummy_img = torch.randn(1, 3, 128, 128, device=device)
    dummy_joint = torch.randn(1, 4, device=device)
    torch.onnx.export(
        model,
        (dummy_img, dummy_joint),
        onnx_export_path,
        export_params=True,
        opset_version=14,
        input_names=["silhouette_image", "naive_joints"],
        output_names=["delta_joints"],
        dynamic_axes={
            "silhouette_image": {0: "batch_size"},
            "naive_joints": {0: "batch_size"},
            "delta_joints": {0: "batch_size"}
        }
    )
    print(f"Exported ultra-lightweight ONNX policy to '{onnx_export_path}' (Ready for Jetson Nano TensorRT / OnnxRuntime)!")

if __name__ == "__main__":
    train_delta_network()
