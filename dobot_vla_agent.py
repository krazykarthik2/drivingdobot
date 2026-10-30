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
    def select_action(self, rgb_image: np.ndarray, robot_state: np.ndarray) -> np.ndarray:
        """
        rgb_image: (H, W, 3) uint8 image in range [0, 255]
        robot_state: (6,) float numpy array (end-effector [x, y, z, roll, pitch, yaw] or joints)
        
        Returns:
            np.ndarray of shape (6,): [dx, dy, dz, droll, dpitch, gripper]
        """
        # Preprocess camera image: resize to 256x256, normalize to [0.0, 1.0], shape (1, 3, 256, 256)
        img_resized = cv2.resize(rgb_image, (256, 256))
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        img_tensor = img_tensor.to(self.device)

        # Low-dimensional robot state: shape (1, 6)
        state_tensor = torch.from_numpy(robot_state.astype(np.float32)).unsqueeze(0).to(self.device)

        # Assemble full batch dictionary conforming to SmolVLA multimodal specifications
        batch = {
            "observation.state": state_tensor,
            "observation.images.camera1": img_tensor,
            "observation.images.camera2": img_tensor,
            "observation.images.camera3": img_tensor,
            "observation.language.tokens": self.input_ids,
            "observation.language.attention_mask": self.attention_mask,
        }

        # Predict physical action step via flow matching
        action_tensor = self.policy.select_action(batch)
        if isinstance(action_tensor, torch.Tensor):
            action = action_tensor.detach().cpu().numpy().squeeze()
        else:
            action = np.array(action_tensor).squeeze()

        return action
