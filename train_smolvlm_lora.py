import os
import sys

# Suppress conflicting frameworks
sys.modules['jax'] = None
sys.modules['jaxlib'] = None
sys.modules['tensorflow'] = None
os.environ['HF_HUB_OFFLINE'] = '1'

import json
import argparse
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, AutoModelForVision2Seq
from peft import LoraConfig, get_peft_model


class SimVLMDataset(Dataset):
    def __init__(self, annotation_path, img_dir, processor, mode="cot"):
        """
        mode: 'cot' (short chain-of-thought) or 'direct' (direct structured tokens)
        """
        self.img_dir = img_dir
        self.processor = processor
        self.mode = mode

        with open(annotation_path, "r") as f:
            self.samples = json.load(f)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        img_path = os.path.join(self.img_dir, item["image"])
        image = Image.open(img_path).convert("RGB")

        target_text = item["cot_response"] if self.mode == "cot" else item["direct_response"]
        user_query = item["instruction"]

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_query}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": target_text}
                ]
            }
        ]

        text = self.processor.apply_chat_template(messages, add_generation_prompt=False)
        inputs = self.processor(text=text, images=[image], return_tensors="pt")

        input_ids = inputs["input_ids"][0]
        labels = input_ids.clone()

        # Mask user prompt tokens so loss is computed ONLY on assistant responses
        prompt_text = self.processor.apply_chat_template(messages[:1], add_generation_prompt=True)
        prompt_ids = self.processor.tokenizer(prompt_text, return_tensors="pt")["input_ids"][0]
        prompt_len = len(prompt_ids)

        labels[:prompt_len] = -100

        item_dict = {k: v[0] for k, v in inputs.items()}
        item_dict["labels"] = labels
        return item_dict


def collate_fn(batch):
    # Padding collation for vision-language inputs
    input_ids = [b["input_ids"] for b in batch]
    labels = [b["labels"] for b in batch]
    pixel_values = [b["pixel_values"] for b in batch]

    # Pad 1D tensors
    padded_input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    padded_labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    pixel_values = torch.stack(pixel_values)

    return {
        "input_ids": padded_input_ids,
        "labels": padded_labels,
        "pixel_values": pixel_values,
        "attention_mask": (padded_input_ids != 0).long()
    }


def train_smolvlm_lora(mode="cot", max_steps=4, batch_size=1, lr=2e-4, out_dir="smolvlm_lora_cot"):
    print("=" * 80)
    print(f"TRAINING SMOLVLM LORA | Mode = '{mode.upper()}' | Output: {out_dir}")
    print("=" * 80)
    sys.stdout.flush()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model_path = r"C:\Users\karthikkrazy\.cache\huggingface\hub\models--HuggingFaceTB--SmolVLM-256M-Instruct\snapshots\7e3e67edbbed1bf9888184d9df282b700a323964"
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForVision2Seq.from_pretrained(model_path, torch_dtype=torch.float32)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.to(device)
    model.train()
    model.print_trainable_parameters()

    dataset = SimVLMDataset(
        annotation_path="harvested_dataset/dataset_annotations.json",
        img_dir="harvested_dataset/images",
        processor=processor,
        mode=mode
    )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    step = 0
    total_loss = 0.0

    for epoch in range(10):
        for batch in dataloader:
            step += 1
            b = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad()
            outputs = model(**b)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            print(f"  Step {step}/{max_steps} | Loss: {loss.item():.4f}")
            sys.stdout.flush()

            if step >= max_steps:
                break
        if step >= max_steps:
            break

    # Save adapter
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    processor.save_pretrained(out_dir)
    print(f"\n[SUCCESS]: Saved {mode.upper()} LoRA adapter to {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["cot", "direct"], default="cot")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--out_dir", type=str, default="smolvlm_lora_cot")
    args = parser.parse_args()

    train_smolvlm_lora(mode=args.mode, max_steps=args.steps, out_dir=args.out_dir)
