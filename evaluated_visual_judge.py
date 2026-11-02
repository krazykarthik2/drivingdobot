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
        """
        Plans robotic subtasks with in-context few-shot guidance to enforce
        invariance, correct reasoning, and dynamic stage counts instead of overfitting to hotspots.
        """
        pil_img = Image.fromarray(rgb_frame.astype(np.uint8))
        
        system_instruction = (
            "You are a robotic task planner. Decompose the user's high-level goal into sequential subtasks.\n"
            "Rules:\n"
            "- 'approach <object>' -> only 2 stages: hover above object, descend near object. NO gripping.\n"
            "- 'push <object> <direction>' -> 4 stages: hover beside, descend, push sweep, retract.\n"
            "- 'pick <object> and place on <target>' -> 7 stages: hover, descend, grip, lift, transfer, lower, release.\n\n"
            "Example 1:\n"
            "Goal: 'approach the pen'\n"
            "<thought>Instruction is only to approach the pen. Zero gripping or lifting required.</thought>\n"
            "<plan>\n"
            "1. hover above pen\n"
            "2. approach near pen\n"
            "</plan>\n\n"
            "Example 2:\n"
            "Goal: 'push the blue pen to the right'\n"
            "<thought>Pushing object requires lateral sweeping without grasping.</thought>\n"
            "<plan>\n"
            "1. hover beside blue pen\n"
            "2. descend to contact height\n"
            "3. push blue pen to the right\n"
            "4. retract arm\n"
            "</plan>\n\n"
            "Example 3:\n"
            "Goal: 'pick up the green cube and place it on red platform'\n"
            "<thought>Full pick-and-place requires grasp, lift, transfer, and release on destination.</thought>\n"
            "<plan>\n"
            "1. hover above green cube\n"
            "2. descend to green cube\n"
            "3. grip green cube\n"
            "4. lift green cube\n"
            "5. transfer to red platform\n"
            "6. lower onto red platform\n"
            "7. release green cube and retract\n"
            "</plan>"
        )

        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": f"{system_instruction}\n\nGoal: '{user_goal}'. Break into robotic subtasks."}
            ]}
        ]
        text_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=text_prompt, images=[pil_img], return_tensors="pt").to(self.device)
        output_ids = self.model.generate(**inputs, max_new_tokens=180, do_sample=False)
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

        output_ids = self.model.generate(**inputs, max_new_tokens=50 if self.mode == "cot" else 10)
        raw_res = self.processor.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

        # In direct mode without thoughts, we can check position-0 logits
        # In CoT mode, the answer is inside <decision>YES/NO</decision> after thought tokens.
        import re
        dec_match = re.search(r'<decision>\s*(YES|NO)\s*</decision>', raw_res, re.IGNORECASE)
        if dec_match:
            is_yes = (dec_match.group(1).upper() == "YES")
            prob_yes = 1.0 if is_yes else 0.0
        elif "yes" in raw_res.lower() and "no" not in raw_res.lower():
            is_yes = True
            prob_yes = 1.0
        elif "no" in raw_res.lower() and "yes" not in raw_res.lower():
            is_yes = False
            prob_yes = 0.0
        else:
            # Fallback to token logits
            outputs = self.model(**inputs)
            next_token_logits = outputs.logits[0, -1, :]
            yes_tokens = [self.processor.tokenizer.encode(t, add_special_tokens=False)[-1] for t in ["Yes", "yes", "YES"]]
            no_tokens = [self.processor.tokenizer.encode(t, add_special_tokens=False)[-1] for t in ["No", "no", "NO"]]
            yes_score = max(next_token_logits[tid].item() for tid in yes_tokens)
            no_score = max(next_token_logits[tid].item() for tid in no_tokens)
            exp_yes = np.exp(yes_score - max(yes_score, no_score))
            exp_no = np.exp(no_score - max(yes_score, no_score))
            prob_yes = exp_yes / (exp_yes + exp_no)
            is_yes = bool(prob_yes > 0.50)

        formatted_res = f"Decision={is_yes} | P(Yes)={prob_yes:.2f} | Raw='{raw_res}'"

        return formatted_res, is_yes

    @torch.no_grad()
    @torch.no_grad()
    def evaluate_stage_with_logits(self, rgb_frame, query):
        """
        Stage gating evaluation:
        In CoT mode, the fine-tuned model outputs thoughts and tags: <thought>...</thought><decision>YES|NO</decision>.
        In Direct mode, it outputs YES or NO.
        We generate the full text and extract the decision, while computing P(Yes) based on the output and logits.
        """
        pil_img = Image.fromarray(rgb_frame.astype(np.uint8))
        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": query}]}
        ]
        text_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=text_prompt, images=[pil_img], return_tensors="pt").to(self.device)

        # Generate response (up to 50 tokens for CoT to complete <decision>)
        max_tokens = 50 if self.mode == "cot" else 10
        output_ids = self.model.generate(**inputs, max_new_tokens=max_tokens)
        gen_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        raw_res = self.processor.decode(gen_tokens, skip_special_tokens=True).strip()

        # Parse decision from generated text
        passed = False
        import re
        dec_match = re.search(r'<decision>\s*(YES|NO)\s*</decision>', raw_res, re.IGNORECASE)
        if dec_match:
            passed = (dec_match.group(1).upper() == "YES")
        elif "yes" in raw_res.lower() and "no" not in raw_res.lower():
            passed = True
        elif "no" in raw_res.lower() and "yes" not in raw_res.lower():
            passed = False
        else:
            # Fallback to direct substring search or logit check if ambiguous
            passed = "yes" in raw_res.lower()

        prob_yes = 1.0 if passed else 0.0
        ans = "Yes" if passed else "No"
        return ans, prob_yes, passed, raw_res
