import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralDeltaDriver(nn.Module):
    """
    System 0B: The Neural Delta Driver (The ML Booster).
    A tiny, ultra-fast Convolutional Neural Network designed to run effortlessly on Jetson Nano.
    
    Inputs:
      - Silhouette image: [B, 3, 128, 128] solid colored shapes (Green=pick, Blue=place, Red=avoid) on black.
      - Naive Joint state / plan: [B, 4] joint angles (j1, j2, j3, j4) from the Programmed Solver.
    
    Output:
      - Delta Joint corrections: [B, 4] (dj1, dj2, dj3, dj4)
        e.g., corrective joint changes to arc over obstacles or twist the wrist for oriented shapes.
    """
    def __init__(self):
        super(NeuralDeltaDriver, self).__init__()
        
        # 3-layer lightweight CNN for silhouette feature extraction
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2) # 128 -> 64
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1) # 64 -> 32
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1) # 32 -> 16
        self.bn3 = nn.BatchNorm2d(64)
        
        self.pool = nn.AdaptiveAvgPool2d((4, 4)) # 64 * 4 * 4 = 1024
        
        # Fusion and Delta Head
        self.fc_vision = nn.Linear(1024, 64)
        self.fc_joint = nn.Linear(4, 32)
        
        self.fc1 = nn.Linear(64 + 32, 64)
        self.fc2 = nn.Linear(64, 32)
        self.delta_head = nn.Linear(32, 4)

    def forward(self, img_silhouettes, naive_joints):
        """
        img_silhouettes: [B, 3, 128, 128] normalized in [0, 1]
        naive_joints: [B, 4] float
        """
        x = F.relu(self.bn1(self.conv1(img_silhouettes)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        feat_vis = F.relu(self.fc_vision(x.view(x.size(0), -1)))
        
        feat_j = F.relu(self.fc_joint(naive_joints))
        
        fused = torch.cat([feat_vis, feat_j], dim=1)
        h = F.relu(self.fc1(fused))
        h = F.relu(self.fc2(h))
        delta = self.delta_head(h)
        return delta

if __name__ == "__main__":
    model = NeuralDeltaDriver()
    print("Neural Delta Driver Parameter Count:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    # Test pass
    dummy_img = torch.zeros(2, 3, 128, 128)
    dummy_joints = torch.zeros(2, 4)
    out = model(dummy_img, dummy_joints)
    print("Output Delta Shape:", out.shape)
