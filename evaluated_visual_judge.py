import os
import sys

# Suppress conflicting frameworks
sys.modules['jax'] = None
sys.modules['jaxlib'] = None
sys.modules['tensorflow'] = None
os.environ['HF_HUB_OFFLINE'] = '1'

import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel


class EvaluatedVisualJudge:
    def __init__(self, mode="base", adapter_dir=None, device="cpu"):
        """
        mode: 'cot', 'direct', or 'base'
        adapter_dir: path to LoRA adapter weights (if mode != 'base')
        """
        self.mode = mode
        self.device = device
        print(f"\n[EvaluatedVisualJudge] Initializing Judge in mode='{self.mode.upper()}' on {self.device}...")

        base_model_path = r"C:\Users\karthikkrazy\.cache\huggingface\hub\models--HuggingFaceTB--SmolVLM-256M-Instruct\snapshots\7e3e67edbbed1bf9888184d9df282b700a323964"
        self.processor = AutoProcessor.from_pretrained(base_model_path)
        base_model = AutoModelForImageTextToText.from_pretrained(base_model_path, torch_dtype=torch.float32)

        if self.mode != "base" and adapter_dir and os.path.exists(adapter_dir):
            print(f"[EvaluatedVisualJudge] Attaching LoRA adapter from: {adapter_dir}")
            self.model = PeftModel.from_pretrained(base_model, adapter_dir)
        else:
            print("[EvaluatedVisualJudge] Using unmodified Base SmolVLM-256M.")
            self.model = base_model

        self.model.to(self.device)
        self.model.eval()
        print(f"[EvaluatedVisualJudge] Ready in mode='{self.mode.upper()}'!\n")

    @torch.no_grad()
    def plan_subtasks(self, rgb_frame, user_goal):
        pil_img = Image.fromarray(rgb_frame.astype(np.uint8))
        if self.mode == "cot":
            prompt = f"Goal: '{user_goal}'. Break into robotic subtasks."
        elif self.mode == "direct":
            prompt = f"Goal: '{user_goal}'. Break into robotic subtasks."
        else:
            prompt = f"User Goal: '{user_goal}'. Break this into sequential robotic subtasks as a concise numbered list."

        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}
        ]
        text_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=text_prompt, images=[pil_img], return_tensors="pt").to(self.device)
        output_ids = self.model.generate(**inputs, max_new_tokens=90)
        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        raw_res = self.processor.decode(generated_ids, skip_special_tokens=True).strip()
        return raw_res

    @torch.no_grad()
    def verify_milestone(self, rgb_frame, query):
        pil_img = Image.fromarray(rgb_frame.astype(np.uint8))
        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": query}]}
        ]
        text_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=text_prompt, images=[pil_img], return_tensors="pt").to(self.device)
        output_ids = self.model.generate(**inputs, max_new_tokens=60 if self.mode == "cot" else 20)
        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        raw_res = self.processor.decode(generated_ids, skip_special_tokens=True).strip()

        # Decision extraction
        res_lower = raw_res.lower()
        if "<decision>yes</decision>" in res_lower:
            is_yes = True
        elif "<decision>no</decision>" in res_lower:
            is_yes = False
        elif "inside" in res_lower or "on the tray" in res_lower:
            # Model directly confirmed semantic containment
            is_yes = True
        elif "yes" in res_lower:
            is_yes = True
        elif "no" in res_lower:
            is_yes = False
        else:
            is_yes = False

        return raw_res, is_yes
