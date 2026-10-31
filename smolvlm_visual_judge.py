"""
smolvlm_visual_judge.py

Autonomous Visual Completion Judge & High-Level Task Reasoner powered by SmolVLM.
Evaluates physical camera observations to:
1. Identify realistic tabletop objects (e.g., books, mugs, trays, notepads)
2. Determine semantic completion states (e.g., "Is arm aligned above target?",
   "Is object grasped?", "Is object placed in target tray?")
3. Dynamically switch low-level VLA subtasks without hardcoded step counts or simulation ground-truth cheats.
"""

import os
import sys
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq


class SmolVLMVisualJudge:
    def __init__(self, device="cpu"):
        self.device = torch.device(device)
        print(f"[SmolVLMVisualJudge] Initializing offline VLM Judge on {self.device}...")

        # Locate cached local snapshot
        cache_base = os.path.expanduser("~/.cache/huggingface/hub/models--HuggingFaceTB--SmolVLM-256M-Instruct/snapshots")
        if os.path.exists(cache_base) and len(os.listdir(cache_base)) > 0:
            snap_dir = os.path.join(cache_base, os.listdir(cache_base)[0])
        else:
            snap_dir = "HuggingFaceTB/SmolVLM-256M-Instruct"

        print(f"[SmolVLMVisualJudge] Loading model from: {snap_dir}")
        self.processor = AutoProcessor.from_pretrained(snap_dir, local_files_only=True)
        self.model = AutoModelForVision2Seq.from_pretrained(
            snap_dir,
            local_files_only=True,
            torch_dtype=torch.float32
        )
        self.model.to(self.device)
        self.model.eval()
        print("[SmolVLMVisualJudge] Offline SmolVLM Judge ready!")

    @torch.no_grad()
    def ask_vlm(self, rgb_image: np.ndarray, query: str, max_new_tokens: int = 15) -> str:
        """Queries SmolVLM with the current camera frame and returns text response."""
        pil_img = Image.fromarray(rgb_image.astype(np.uint8))
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": query}
                ]
            }
        ]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=pil_img, return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        decoded = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        # Extract assistant turn
        if "Assistant:" in decoded:
            ans = decoded.split("Assistant:")[-1].strip()
        else:
            ans = decoded.strip()
        return ans

    def judge_phase_completion(self, rgb_image: np.ndarray, phase: str, target_name: str) -> tuple:
        """
        Visually judges whether a specific phase has completed.
        Returns: (is_completed: bool, reason: str)
        """
        if phase == "approach":
            query = f"Look closely at the robot gripper and the {target_name}. Is the robot gripper positioned directly above or near the {target_name}? Answer with YES or NO."
        elif phase == "grasp":
            query = f"Look at the robot gripper fingertips. Is the {target_name} held or grasped between the gripper fingers? Answer with YES or NO."
        elif phase == "place":
            query = f"Is the {target_name} resting securely on or inside the destination? Answer with YES or NO."
        else:
            query = f"Has the task '{phase}' succeeded? Answer with YES or NO."

        ans = self.ask_vlm(rgb_image, query)
        is_yes = "yes" in ans.lower()
        return is_yes, ans
