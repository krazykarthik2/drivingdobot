"""
Dobot 3D DP3 Policy Controller with Canonical Point Cloud Alignment.
Directly loads the official 3D Diffusion Policy (DP3) model trained on bin-picking
from `dp3_official/latest_binpicking.ckpt`.
Projects 3D Point Clouds and robot state into the policy's canonical reference frame,
runs the 255M parameter diffusion policy model step-by-step,
and maps the predicted actions back to Dobot control without any oracle coordinates or hardcoded waypoints.
"""
import os
import sys
import numpy as np
import torch
import dill
from omegaconf import OmegaConf
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

# Add 3D-Diffusion-Policy to Python path
dp3_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "dp3_official", "3D-Diffusion-Policy"))
if dp3_root not in sys.path:
    sys.path.append(dp3_root)

from diffusion_policy_3d.policy.dp3 import DP3

class DobotDP3Controller:
    def __init__(self, ckpt_path="dp3_official/latest_binpicking.ckpt", device="cpu"):
        self.device = torch.device(device)
        print(f"[DobotDP3Controller] Loading official 3D DP3 weights from {ckpt_path} on {device}...")
        payload = torch.load(ckpt_path, map_location=self.device, pickle_module=dill, weights_only=False)
        cfg = payload["cfg"]

        policy_cfg = OmegaConf.to_container(cfg.policy, resolve=True)
        shape_meta = OmegaConf.to_container(cfg.policy.shape_meta, resolve=True)

        scheduler = DDIMScheduler(
            num_train_timesteps=policy_cfg["noise_scheduler"]["num_train_timesteps"],
            beta_start=policy_cfg["noise_scheduler"]["beta_start"],
            beta_end=policy_cfg["noise_scheduler"]["beta_end"],
            beta_schedule=policy_cfg["noise_scheduler"]["beta_schedule"],
            clip_sample=policy_cfg["noise_scheduler"]["clip_sample"],
            set_alpha_to_one=policy_cfg["noise_scheduler"]["set_alpha_to_one"],
            steps_offset=policy_cfg["noise_scheduler"]["steps_offset"],
            prediction_type=policy_cfg["noise_scheduler"]["prediction_type"]
        )

        encoder_cfg = OmegaConf.create({
            "in_channels": 3,
            "out_channels": policy_cfg["encoder_output_dim"],
            "use_layernorm": True,
            "final_norm": "layernorm",
            "normal_channel": False
        })

        self.policy = DP3(
            shape_meta=shape_meta,
            noise_scheduler=scheduler,
            horizon=policy_cfg["horizon"],
            n_action_steps=policy_cfg["n_action_steps"],
            n_obs_steps=policy_cfg["n_obs_steps"],
            num_inference_steps=policy_cfg["num_inference_steps"],
            obs_as_global_cond=policy_cfg["obs_as_global_cond"],
            diffusion_step_embed_dim=policy_cfg["diffusion_step_embed_dim"],
            down_dims=policy_cfg["down_dims"],
            kernel_size=policy_cfg["kernel_size"],
            n_groups=policy_cfg["n_groups"],
            condition_type=policy_cfg["condition_type"],
            use_down_condition=policy_cfg["use_down_condition"],
            use_mid_condition=policy_cfg["use_mid_condition"],
            use_up_condition=policy_cfg["use_up_condition"],
            encoder_output_dim=policy_cfg["encoder_output_dim"],
            use_pc_color=policy_cfg["use_pc_color"],
            pointnet_type=policy_cfg["pointnet_type"],
            pointcloud_encoder_cfg=encoder_cfg,
            use_point_crop=policy_cfg["use_point_crop"]
        ).to(self.device)

        self.policy.load_state_dict(payload["state_dicts"]["model"])
        self.policy.eval()

        self.n_obs_steps = policy_cfg["n_obs_steps"]
        self.n_action_steps = policy_cfg["n_action_steps"]
        self.obs_history = []
        print("[DobotDP3Controller] 3D DP3 Policy initialized successfully!")

    def reset(self):
        self.obs_history = []

    def to_canonical_frame(self, pts):
        """
        Maps Dobot physical coordinates to the DP3 model's training canonical frame:
        Dobot forward (x: 0.17->0.27) -> Canonical y (0.60->0.75)
        Dobot lateral (y: -0.10->0.10) -> Canonical x (-0.15->0.15)
        Dobot vertical (z: -0.04->0.08) -> Canonical z (0.06->0.20)
        """
        p = np.copy(pts)
        x_c = p[..., 1]
        y_c = 0.68 + (p[..., 0] - 0.22) * 1.2
        z_c = 0.11 + (p[..., 2] - 0.02) * 1.0
        return np.stack([x_c, y_c, z_c], axis=-1).astype(np.float32)

    def actions_to_dobot(self, act_canonical):
        """
        Maps canonical action delta [dx_c, dy_c, dz_c, grip] back to Dobot delta [dx, dy, dz, grip]:
        dx_dobot = dy_c / 1.2
        dy_dobot = dx_c
        dz_dobot = dz_c
        """
        dx_c = act_canonical[:, 0]
        dy_c = act_canonical[:, 1]
        dz_c = act_canonical[:, 2]
        grip = act_canonical[:, 3]

        dx = dy_c / 1.2
        dy = dx_c
        dz = dz_c
        return np.stack([dx, dy, dz, grip], axis=-1)

    def predict(self, raw_point_cloud_512x3, raw_agent_pos_9d):
        """
        Inputs:
            raw_point_cloud_512x3: (512, 3) raw metric point cloud from camera
            raw_agent_pos_9d: (9,) [ee_tip (3), finger_right (3), finger_left (3)]
        Returns:
            actions: (n_action_steps, 4) Dobot Cartesian deltas [dx, dy, dz, grip]
        """
        pc_canonical = self.to_canonical_frame(raw_point_cloud_512x3)

        ee = self.to_canonical_frame(raw_agent_pos_9d[:3])
        fr = self.to_canonical_frame(raw_agent_pos_9d[3:6])
        fl = self.to_canonical_frame(raw_agent_pos_9d[6:9])
        agent_pos_canonical = np.concatenate([ee, fr, fl]).astype(np.float32)

        obs_item = {
            "point_cloud": torch.from_numpy(pc_canonical).float(),
            "agent_pos": torch.from_numpy(agent_pos_canonical).float()
        }

        if len(self.obs_history) == 0:
            for _ in range(self.n_obs_steps):
                self.obs_history.append(obs_item)
        else:
            self.obs_history.append(obs_item)
            if len(self.obs_history) > self.n_obs_steps:
                self.obs_history.pop(0)

        batch_pc = torch.stack([x["point_cloud"] for x in self.obs_history], dim=0).unsqueeze(0).to(self.device)
        batch_pos = torch.stack([x["agent_pos"] for x in self.obs_history], dim=0).unsqueeze(0).to(self.device)

        obs_dict = {
            "point_cloud": batch_pc,
            "agent_pos": batch_pos
        }

        with torch.no_grad():
            res = self.policy.predict_action(obs_dict)
            acts_c = res["action"][0].cpu().numpy()

        acts_dobot = self.actions_to_dobot(acts_c[:self.n_action_steps])
        return acts_dobot
