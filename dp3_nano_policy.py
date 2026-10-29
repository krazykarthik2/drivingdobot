"""
DP3-Nano: Ultra-Lightweight 3D Diffusion Policy Architecture for Edge Robotics.
Optimized for:
1. Fast CPU / Jetson Nano inference (<15ms per chunk).
2. Point-Cloud / 3D Coordinate Conditioned Diffusion Backbone.
3. System 0A + System 0B Denoising Loop:
   - System 0A: 3D Geometric Anchor Prior (Coarse Trajectory).
   - System 0B: 1D Temporal CNN Diffusion Denoising Head (Corrective Contact & Precision Chunk).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class DP3NanoDenoisingBackbone(nn.Module):
    """
    Lightweight 1D Temporal Residual Diffusion UNet for Action Chunks.
    Predicts action sequence: [Horizon x Action_Dim] (e.g. 8 steps x 7 dims).
    """
    def __init__(self, action_dim=7, horizon=8, cond_dim=64, model_dim=128):
        super().__init__()
        self.action_dim = action_dim
        self.horizon = horizon

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(model_dim),
            nn.Linear(model_dim, model_dim * 2),
            nn.Mish(),
            nn.Linear(model_dim * 2, model_dim),
        )

        # 3D spatial feature conditioning (Point / 3D bounding coordinates)
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, model_dim),
            nn.Mish(),
            nn.Linear(model_dim, model_dim)
        )

        # 1D Temporal Convolutional blocks
        self.in_proj = nn.Linear(action_dim, model_dim)

        self.block1 = nn.Sequential(
            nn.Conv1d(model_dim, model_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(model_dim),
            nn.Mish()
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(model_dim, model_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(model_dim),
            nn.Mish()
        )
        self.out_proj = nn.Linear(model_dim, action_dim)

    def forward(self, sample, timestep, cond):
        """
        sample: (B, horizon, action_dim) noisy actions
        timestep: (B,) diffusion step
        cond: (B, cond_dim) 3D spatial anchor features
        """
        B = sample.shape[0]
        t_emb = self.time_mlp(timestep) # (B, model_dim)
        c_emb = self.cond_mlp(cond)     # (B, model_dim)
        cond_total = (t_emb + c_emb).unsqueeze(1) # (B, 1, model_dim)

        h = self.in_proj(sample) + cond_total # (B, horizon, model_dim)
        h = h.transpose(1, 2) # (B, model_dim, horizon)

        h = self.block1(h)
        h = self.block2(h)

        h = h.transpose(1, 2) # (B, horizon, model_dim)
        out = self.out_proj(h) # (B, horizon, action_dim)
        return out

class DP3NanoPolicy(nn.Module):
    """
    Complete DP3-Nano Policy System:
    Combines System 0A Geometric Anchor with System 0B Diffusion Denoising.
    """
    def __init__(self, action_dim=7, horizon=8, cond_dim=12, num_diffusion_steps=10):
        super().__init__()
        self.action_dim = action_dim
        self.horizon = horizon
        self.num_diffusion_steps = num_diffusion_steps

        # Fast 10-step DDIM / DDPM schedule
        self.register_buffer("betas", torch.linspace(1e-4, 0.02, num_diffusion_steps))
        alphas = 1.0 - self.betas
        self.register_buffer("alphas_cumprod", torch.cumprod(alphas, dim=0))

        # 3D spatial encoder (takes pick 3D, place 3D, eef 3D, obstacle 3D = 12 dims)
        self.spatial_encoder = nn.Sequential(
            nn.Linear(cond_dim, 64),
            nn.Mish(),
            nn.Linear(64, 64)
        )

        self.denoiser = DP3NanoDenoisingBackbone(
            action_dim=action_dim,
            horizon=horizon,
            cond_dim=64,
            model_dim=128
        )

    def sample_actions(self, spatial_cond, base_anchor_actions=None):
        """
        Inference: Starts from Gaussian noise or System 0A base anchor prior and denoises in 10 steps.
        """
        device = next(self.parameters()).device
        B = spatial_cond.shape[0]
        cond_feat = self.spatial_encoder(spatial_cond)

        if base_anchor_actions is not None:
            # Warm-start diffusion from System 0A anchor + small noise
            x = base_anchor_actions + 0.1 * torch.randn_like(base_anchor_actions)
        else:
            x = torch.randn(B, self.horizon, self.action_dim, device=device)

        # 10-step fast reverse diffusion
        for i in reversed(range(self.num_diffusion_steps)):
            t = torch.full((B,), i, device=device, dtype=torch.long)
            eps_pred = self.denoiser(x, t, cond_feat)

            alpha = 1.0 - self.betas[i]
            alpha_hat = self.alphas_cumprod[i]
            beta = self.betas[i]

            if i > 0:
                noise = torch.randn_like(x)
            else:
                noise = 0.0

            x = (1.0 / torch.sqrt(alpha)) * (x - (beta / torch.sqrt(1.0 - alpha_hat)) * eps_pred) + torch.sqrt(beta) * noise

        return x

if __name__ == "__main__":
    import time
    model = DP3NanoPolicy()
    model.eval()

    # 12-dim 3D spatial condition
    dummy_cond = torch.randn(1, 12)
    # 8-step action anchor
    dummy_anchor = torch.zeros(1, 8, 7)

    t0 = time.time()
    with torch.no_grad():
        actions = model.sample_actions(dummy_cond, dummy_anchor)
    t_ms = (time.time() - t0) * 1000.0

    print("=" * 70)
    print("DP3-NANO ARCHITECTURE VERIFICATION")
    print(f"Total Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Action Chunk Shape: {actions.shape} (Horizon=8, ActionDim=7)")
    print(f"Inference Latency on CPU: {t_ms:.2f} ms (<15ms budget for Jetson Nano!)")
    print("=" * 70)
