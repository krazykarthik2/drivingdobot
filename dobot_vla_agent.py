"""
dobot_vla_agent.py

End-to-end Vision-Language-Action (VLA) controller powered by Hugging Face's SmolVLA
(lerobot/smolvla_base, 450M parameter flow-matching VLA).

Accepts natural language commands (e.g. "pick up the red cube and place on red plate")
and raw visual observations from the front-mounted 45-degree camera, predicting closed-loop
physical 6D actions for the Dobot Magician manipulator.
"""

import sys
import os
import torch
import numpy as np
import cv2

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


class DobotVLAAgent:
    def __init__(self, pretrained_path="lerobot/smolvla_base", device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = torch.device(device)
        print(f"[DobotVLAAgent] Initializing pre-trained SmolVLA policy on {self.device}...")

        # Load Tokenizer
        print("[DobotVLAAgent] Loading SmolVLM2 tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")

        # Load SmolVLA Policy
        print(f"[DobotVLAAgent] Loading SmolVLAPolicy from {pretrained_path}...")
        self.policy = SmolVLAPolicy.from_pretrained(pretrained_path)
        self.policy.to(self.device)
        self.policy.eval()

        # SO-100 unnormalization statistics from LeRobot smolvla_base checkpoint
        self.action_mean = np.array([1.5959, 119.9442, 109.7698, 56.7055, -27.4227, 12.0028], dtype=np.float32)
        self.action_std = np.array([26.3918, 52.4114, 49.8538, 36.9983, 59.3600, 19.0400], dtype=np.float32)

        self.instruction = ""
        self.input_ids = None
        self.attention_mask = None
        print("[DobotVLAAgent] Pre-trained VLA Policy ready!")

    def set_task_instruction(self, text_instruction: str):
        """Sets active natural language goal and resets internal policy recurrent/kv caches."""
        self.instruction = text_instruction
        self.policy.reset()

        tokens = self.tokenizer(
            text_instruction,
            return_tensors="pt",
            padding="max_length",
            max_length=64,
            truncation=True
        )
        self.input_ids = tokens["input_ids"].to(self.device)
        self.attention_mask = tokens["attention_mask"].to(self.device)
        print(f"[DobotVLAAgent] Active Task Instruction: '{text_instruction}'")

    @torch.no_grad()
    def select_action(self, rgb_image: np.ndarray, robot_state: np.ndarray, wrist_image: np.ndarray = None, side_image: np.ndarray = None, temperature: float = 0.0) -> tuple:
        """
        rgb_image: (H, W, 3) front camera view (camera1)
        robot_state: (6,) float robot state
        wrist_image: (H, W, 3) wrist camera view (camera2)
        side_image: (H, W, 3) side camera view (camera3)
        temperature: Flow-matching sampling temperature (0.0 = deterministic ODE flow from mode, <1.0 = low noise)
        
        Returns:
            (action_norm, action_unnorm)
        """
        # Preprocess front camera image (camera1)
        img_resized = cv2.resize(rgb_image, (256, 256))
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        img_tensor = img_tensor.to(self.device)

        # Preprocess wrist camera image (camera2)
        if wrist_image is not None:
            w_resized = cv2.resize(wrist_image, (256, 256))
            wrist_tensor = torch.from_numpy(w_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            wrist_tensor = wrist_tensor.to(self.device)
        else:
            wrist_tensor = img_tensor

        # Preprocess side camera image (camera3)
        if side_image is not None:
            s_resized = cv2.resize(side_image, (256, 256))
            side_tensor = torch.from_numpy(s_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            side_tensor = side_tensor.to(self.device)
        else:
            side_tensor = img_tensor

        # Low-dimensional robot state: shape (1, 6)
        state_tensor = torch.from_numpy(robot_state.astype(np.float32)).unsqueeze(0).to(self.device)

        # Assemble full batch dictionary conforming to SmolVLA multimodal specifications:
        # camera1: Front elevated view
        # camera2: Eye-in-hand Wrist view
        # camera3: Lateral Side profile view
        batch = {
            "observation.state": state_tensor,
            "observation.images.camera1": img_tensor,
            "observation.images.camera2": wrist_tensor,
            "observation.images.camera3": side_tensor,
            "observation.language.tokens": self.input_ids,
            "observation.language.attention_mask": self.attention_mask,
        }

        # Temperature-scaled initial noise for flow-matching ODE:
        # temperature = 0.0 -> deterministic zero-noise ODE integration (maximum mode coherence)
        # temperature < 1.0 -> reduced variance sampling
        if temperature <= 1e-4:
            noise = torch.zeros(
                1, self.policy.config.chunk_size, self.policy.config.max_action_dim,
                device=self.device, dtype=self.policy.dtype
            )
        else:
            noise = torch.randn(
                1, self.policy.config.chunk_size, self.policy.config.max_action_dim,
                device=self.device, dtype=self.policy.dtype
            ) * temperature

        # Predict physical action step via flow matching
        action_tensor = self.policy.select_action(batch, noise=noise)
        if isinstance(action_tensor, torch.Tensor):
            action_norm = action_tensor.detach().cpu().numpy().squeeze()
        else:
            action_norm = np.array(action_tensor).squeeze()

        # Unnormalize model output to physical SO-100 joint space (degrees)
        action_unnorm = action_norm * self.action_std + self.action_mean

        return action_norm, action_unnorm
