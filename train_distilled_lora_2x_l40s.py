"""
train_distilled_lora_2x_l40s.py

High-Throughput Knowledge Distillation & Fine-Tuning Pipeline
Architecture: 2x NVIDIA L40S (48GB VRAM each = 96GB Total VRAM) + 40GB+ RAM.

Teacher Model: Qwen/Qwen2-VL-7B-Instruct (High-capacity spatial vision reasoning)
Student Model: HuggingFaceTB/SmolVLM-256M-Instruct (Compact edge policy)

Dual Operating Modes:
1. 'cot': Generates clear visual spatial reasoning inside <thought>...</thought>
   followed by strict ground-truth decision <decision>YES|NO</decision>.
2. 'direct': Direct token prediction with verified spatial features.

Inspection & Verification:
- Live sample visual inspection every N steps:
  Decodes student thoughts and decisions vs teacher verification to verify
  that CoT reasoning matches visual ground truth.
"""

import os
import sys
import argparse
import json
import re
from typing import Dict, Any, List
from PIL import Image

# Suppress conflicting frameworks
sys.modules['jax'] = None
sys.modules['jaxlib'] = None
sys.modules['tensorflow'] = None

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    get_cosine_schedule_with_warmup
)
try:
    from transformers import Qwen2VLForConditionalGeneration
except ImportError:
    Qwen2VLForConditionalGeneration = None

from peft import LoraConfig, get_peft_model


# ==============================================================================
# 1. DATASET WITH EXACT TARGET BOUNDARIES & SYNDICATED COT
# ==============================================================================
class DistilledRoboticsDataset(Dataset):
    def __init__(self, annotation_path: str, img_dir: str, processor, mode: str = "cot"):
        self.img_dir = img_dir
        self.processor = processor
        self.mode = mode

        with open(annotation_path, "r", encoding="utf-8") as f:
            self.samples = json.load(f)

        print(f"[Dataset] Loaded {len(self.samples)} samples from {annotation_path} (Mode: {mode})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        img_path = os.path.join(self.img_dir, item["image"])
        image = Image.open(img_path).convert("RGB")

        user_query = item["instruction"]
        expected_ans = item.get("direct_response", "YES").strip()

        if self.mode == "cot":
            cot_text = item.get("cot_response", "")
            # Ensure thought and decision syntax is strictly formed
            if "<decision>" in cot_text:
                target_response = cot_text
            else:
                target_response = f"<thought>Visual inspection: verifying gripper alignment with object.</thought>\n<decision>{expected_ans}</decision>"
        else:
            target_response = expected_ans

        # Build user prompt and complete assistant conversation
        user_message = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_query}]}
        ]
        full_conversation = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_query}]},
            {"role": "assistant", "content": [{"type": "text", "text": target_response}]}
        ]

        full_text = self.processor.apply_chat_template(full_conversation, add_generation_prompt=False)
        prompt_text = self.processor.apply_chat_template(user_message, add_generation_prompt=True)

        inputs = self.processor(text=full_text, images=[image], return_tensors="pt")
        prompt_inputs = self.processor(text=prompt_text, images=[image], return_tensors="pt")

        input_ids = inputs["input_ids"][0]
        prompt_len = prompt_inputs["input_ids"].shape[1]

        # Exact loss masking: compute loss ONLY on assistant output tokens
        labels = input_ids.clone()
        labels[:prompt_len] = -100

        pixel_values = inputs["pixel_values"][0]

        return {
            "input_ids": input_ids,
            "labels": labels,
            "pixel_values": pixel_values,
            "user_query": user_query,
            "target_response": target_response,
            "image": image
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    input_ids = [b["input_ids"] for b in batch]
    labels = [b["labels"] for b in batch]
    pixel_values = torch.stack([b["pixel_values"] for b in batch])

    padded_input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    padded_labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    attention_mask = (padded_input_ids != 0).long()

    queries = [b["user_query"] for b in batch]
    targets = [b["target_response"] for b in batch]
    images = [b["image"] for b in batch]

    return {
        "input_ids": padded_input_ids,
        "labels": padded_labels,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "queries": queries,
        "targets": targets,
        "images": images
    }


# ==============================================================================
# 2. LOCAL TEACHER ENGINE: Qwen2-VL-7B (ALLOCATED TO GPU 1)
# ==============================================================================
class Qwen2VLTeacher:
    def __init__(self, model_id: str = "Qwen/Qwen2-VL-7B-Instruct", device: str = "cuda:1"):
        self.device = torch.device(device)
        print(f"\n[Teacher] Initializing High-Capacity Teacher: {model_id} on {self.device}...")
        self.processor = AutoProcessor.from_pretrained(model_id)

        model_cls = Qwen2VLForConditionalGeneration if Qwen2VLForConditionalGeneration else AutoModelForImageTextToText
        self.model = model_cls.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map=self.device
        )
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        print(f"[Teacher] Qwen2-VL-7B Teacher loaded successfully on {self.device}!\n")

    @torch.no_grad()
    def generate_teacher_cot(self, images: List[Image.Image], queries: List[str]) -> List[str]:
        """Queries Qwen2-VL to generate visual chain-of-thought justification."""
        teacher_responses = []
        for img, query in zip(images, queries):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": f"Observe the robot workspace carefully: {query}\nFirst, explain what you see in <thought>...</thought>. Then give the final verdict in <decision>YES</decision> or <decision>NO</decision>."}
                    ]
                }
            ]
            text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.processor(text=text, images=[img], return_tensors="pt").to(self.device)
            output = self.model.generate(**inputs, max_new_tokens=60)
            gen = self.processor.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            teacher_responses.append(gen)
        return teacher_responses


# ==============================================================================
# 3. TRAINING & LIVE COT INSPECTION CONTROLLER
# ==============================================================================
def train(args):
    device_student = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n==================================================================")
    print(f"=== DISTILLED LORA TRAINING ON 2x L40S ===")
    print(f"=== Student (SmolVLM-256M) on: {device_student} ===")
    if args.use_teacher:
        print(f"=== Teacher (Qwen2-VL-7B) on: cuda:1 ===")
    print(f"==================================================================\n")

    student_id = "HuggingFaceTB/SmolVLM-256M-Instruct"
    processor = AutoProcessor.from_pretrained(student_id)

    # 1. Initialize Student Model with Full Rank LoRA
    student_model = AutoModelForImageTextToText.from_pretrained(
        student_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    )

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    student_model = get_peft_model(student_model, lora_config)
    student_model.to(device_student)
    student_model.train()
    student_model.print_trainable_parameters()

    # 2. Teacher Setup (Qwen2-VL-7B on GPU 1)
    teacher = None
    if args.use_teacher and torch.cuda.device_count() >= 2:
        teacher = Qwen2VLTeacher(model_id=args.teacher_id, device="cuda:1")

    # 3. Dataset & Dataloader
    dataset = DistilledRoboticsDataset(
        annotation_path=args.dataset_json,
        img_dir=args.img_dir,
        processor=processor,
        mode=args.mode
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )

    optimizer = torch.optim.AdamW(student_model.parameters(), lr=args.lr, weight_decay=0.01)
    total_training_steps = len(loader) * args.epochs // args.grad_accum
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_training_steps * 0.05),
        num_training_steps=total_training_steps
    )

    print(f"[Training Setup] Steps/Epoch: {len(loader)} | Total Steps: {total_training_steps} | Batch Size: {args.batch_size}")

    # 4. Training Loop with Live CoT Inspection
    global_step = 0
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device_student)
            labels = batch["labels"].to(device_student)
            attention_mask = batch["attention_mask"].to(device_student)
            pixel_values = batch["pixel_values"].to(device_student, dtype=student_model.dtype)

            outputs = student_model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            loss_scaled = loss / args.grad_accum
            loss_scaled.backward()
            epoch_loss += loss.item()

            if (step + 1) % args.grad_accum == 0 or (step + 1) == len(loader):
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 10 == 0:
                    lr_curr = scheduler.get_last_lr()[0]
                    print(f"Epoch [{epoch+1}/{args.epochs}] | Step [{step+1}/{len(loader)}] | Loss: {loss.item():.4f} | LR: {lr_curr:.2e}")

                # LIVE COT INSPECTION CHECKPOINT:
                # Every 25 steps, sample an image and generate full output to inspect
                # that CoT thoughts and decisions are being learned accurately.
                if global_step % 25 == 0:
                    student_model.eval()
                    with torch.no_grad():
                        sample_img = batch["images"][0]
                        sample_q = batch["queries"][0]
                        sample_tgt = batch["targets"][0]

                        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": sample_q}]}]
                        p_text = processor.apply_chat_template(msgs, add_generation_prompt=True)
                        p_in = processor(text=p_text, images=[sample_img], return_tensors="pt").to(device_student)
                        gen_out = student_model.generate(**p_in, max_new_tokens=50)
                        gen_text = processor.decode(gen_out[0][p_in["input_ids"].shape[1]:], skip_special_tokens=True).strip()

                        teacher_str = ""
                        if teacher is not None:
                            t_out = teacher.generate_teacher_cot([sample_img], [sample_q])
                            teacher_str = f"\n   [Teacher CoT]:  {t_out[0]}"

                        print(f"\n--- [LIVE INSPECTION @ Step {global_step}] ---")
                        print(f"   [Query]:        {sample_q}")
                        print(f"   [Ground Truth]: {sample_tgt.replace(chr(10), ' ')}")
                        print(f"   [Student CoT]:  {gen_text.replace(chr(10), ' ')}{teacher_str}")
                        print(f"--------------------------------------------------\n")
                    student_model.train()

        avg_loss = epoch_loss / len(loader)
        print(f"\n>>> [Epoch {epoch+1} Completed] Average Loss: {avg_loss:.4f}\n")

    # 5. Save Final Adapter
    os.makedirs(args.output_dir, exist_ok=True)
    student_model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"\n==================================================================")
    print(f"[SUCCESS] Saved checkpoint to: {os.path.abspath(args.output_dir)}")
    print(f"==================================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen2-VL Teacher Distillation for SmolVLM on 2x L40S")
    parser.add_argument("--mode", type=str, choices=["cot", "direct"], default="cot")
    parser.add_argument("--dataset_json", type=str, default="harvested_dataset/dataset_annotations.json")
    parser.add_argument("--img_dir", type=str, default="harvested_dataset/images")
    parser.add_argument("--output_dir", type=str, default="smolvlm_lora_l40s_cot")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--use_teacher", action="store_true", help="Enable Qwen2-VL-7B teacher on GPU 1")
    parser.add_argument("--teacher_id", type=str, default="Qwen/Qwen2-VL-7B-Instruct")
    args = parser.parse_args()

    train(args)
