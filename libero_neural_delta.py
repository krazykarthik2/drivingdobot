import torch
import torch.nn as nn
import torch.nn.functional as F

class LiberoNeuralDeltaDriver(nn.Module):
    """
    System 0B: Lightweight Convolutional Residual Delta Driver for LIBERO Franka Panda.
    Inputs:
      - 128x128x3 painted silhouette mask image on black
      - 7D Base Cartesian action [dx, dy, dz, droll, dpitch, dyaw, grasp]
    Output:
      - 7D Corrective delta action [delta_dx, delta_dy, delta_dz, 0, 0, 0, delta_grasp]
    """
    def __init__(self):
        super().__init__()
        # Conv backbone for painted silhouette
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2) # 64x64
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2) # 32x32
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1) # 16x16
        self.pool = nn.AdaptiveAvgPool2d((4, 4)) # 64 * 16 = 1024
        
        self.act_fc = nn.Linear(7, 64)
        
        self.fc1 = nn.Linear(1024 + 64, 256)
        self.fc2 = nn.Linear(256, 128)
        self.out = nn.Linear(128, 7)
        
        # Initialize output layer small so delta starts near zero
        nn.init.uniform_(self.out.weight, -1e-3, 1e-3)
        nn.init.zeros_(self.out.bias)

    def forward(self, img, base_act):
        x_img = F.relu(self.conv1(img))
        x_img = F.relu(self.conv2(x_img))
        x_img = F.relu(self.conv3(x_img))
        x_img = self.pool(x_img).reshape(img.size(0), -1)

        x_act = F.relu(self.act_fc(base_act))

        feat = torch.cat([x_img, x_act], dim=-1)
        x = F.relu(self.fc1(feat))
        x = F.relu(self.fc2(x))
        delta = self.out(x)
        return delta
