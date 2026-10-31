import os
import modal

app = modal.App("smolvlm-robot-lora-tuning")

# Define lightweight container image with GPU support
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "torchvision",
        "transformers>=4.46.0",
        "peft>=0.13.0",
        "accelerate>=0.34.0",
        "pillow",
        "opencv-python-headless"
    )
)

# Persistent volume to store trained LoRA adapter weights
volume = modal.Volume.from_name("smolvlm-adapters-vol", create_if_missing=True)


@app.function(
    image=image,
    gpu="T4", # Or "A10G" / "L4"
    volumes={"/adapters": volume},
    timeout=1800
)
def train_smolvlm_modal(mode: str = "cot", epochs: int = 5, lr: float = 2e-4):
    """
    mode: 'cot' (Short Chain-of-Thought) or 'direct' (Direct Structured Tokens)
    """
    import json
    import torch
    from PIL import Image
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoProcessor, AutoModelForVision2Seq
    from peft import LoraConfig, get_peft_model

    print(f"=== Running SmolVLM LoRA Fine-Tuning | Mode: {mode.upper()} ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    model_id = "HuggingFaceTB/SmolVLM-256M-Instruct"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    )

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

    # Load harvested dataset
    with open("harvested_dataset/dataset_annotations.json", "r") as f:
        samples = json.load(f)

    class FastDataset(Dataset):
        def __init__(self, data):
            self.data = data
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            item = self.data[idx]
            img = Image.open(os.path.join("harvested_dataset/images", item["image"])).convert("RGB")
            target_text = item["cot_response"] if mode == "cot" else item["direct_response"]
            
            messages = [
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": item["instruction"]}]},
                {"role": "assistant", "content": [{"type": "text", "text": target_text}]}
            ]
            text = processor.apply_chat_template(messages, add_generation_prompt=False)
            inputs = processor(text=text, images=[img], return_tensors="pt")
            
            input_ids = inputs["input_ids"][0]
            labels = input_ids.clone()
            prompt_text = processor.apply_chat_template(messages[:1], add_generation_prompt=True)
            prompt_len = len(processor.tokenizer(prompt_text, return_tensors="pt")["input_ids"][0])
            labels[:prompt_len] = -100

            res = {k: v[0] for k, v in inputs.items()}
            res["labels"] = labels
            return res

    def collate(batch):
        input_ids = torch.nn.utils.rnn.pad_sequence([b["input_ids"] for b in batch], batch_first=True, padding_value=0)
        labels = torch.nn.utils.rnn.pad_sequence([b["labels"] for b in batch], batch_first=True, padding_value=-100)
        pixel_values = torch.stack([b["pixel_values"] for b in batch])
        return {
            "input_ids": input_ids,
            "labels": labels,
            "pixel_values": pixel_values,
            "attention_mask": (input_ids != 0).long()
        }

    loader = DataLoader(FastDataset(samples), batch_size=4, shuffle=True, collate_fn=collate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        ep_loss = 0.0
        for step, batch in enumerate(loader):
            b = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            out = model(**b)
            loss = out.loss
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()
        avg = ep_loss / len(loader)
        print(f"Epoch {epoch+1}/{epochs} | Average Loss: {avg:.4f}")

    out_path = f"/adapters/smolvlm_lora_{mode}"
    model.save_pretrained(out_path)
    processor.save_pretrained(out_path)
    volume.commit()
    print(f"Model saved and committed to Modal Volume: {out_path}")
    return f"Done: {out_path}"


@app.local_entrypoint()
def main():
    print("Launching Dual LoRA Fine-Tuning on Modal GPU...")
    print("1. Training Short CoT Adapter...")
    res_cot = train_smolvlm_modal.remote(mode="cot", epochs=5)
    print("2. Training Direct Tokens Adapter...")
    res_dir = train_smolvlm_modal.remote(mode="direct", epochs=5)
    print(f"\nAll complete!\nCoT: {res_cot}\nDirect: {res_dir}")
