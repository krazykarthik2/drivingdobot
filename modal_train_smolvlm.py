import os
import modal

app = modal.App("smolvlm-robot-lora-tuning")

# Container image with PyTorch GPU, PEFT, and Transformers, plus local harvested dataset
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
    .add_local_dir(
        local_path=os.path.abspath("harvested_dataset"),
        remote_path="/root/harvested_dataset"
    )
)

# Modal Volume to persist trained adapters
volume = modal.Volume.from_name("smolvlm-adapters-vol", create_if_missing=True)


@app.function(
    image=image,
    gpu="T4",  # Lowest cost hardware on Modal (~$0.59/hr, ~0.02 USD total cost)
    volumes={"/adapters": volume},
    timeout=1200
)
def train_smolvlm_modal(mode: str = "cot", epochs: int = 3, lr: float = 2e-4):
    """
    mode: 'cot' (Short Chain-of-Thought) or 'direct' (Direct Structured Tokens)
    """
    import json
    import torch
    from PIL import Image
    from torch.utils.data import Dataset, DataLoader
    try:
        from transformers import AutoModelForImageTextToText as VisionModelClass
    except ImportError:
        from transformers import AutoModelForVision2Seq as VisionModelClass
    from transformers import AutoProcessor
    from peft import LoraConfig, get_peft_model

    print(f"\n=======================================================")
    print(f"=== Fine-Tuning SmolVLM on Modal GPU | Mode: {mode.upper()} ===")
    print(f"=======================================================")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    model_id = "HuggingFaceTB/SmolVLM-256M-Instruct"
    processor = AutoProcessor.from_pretrained(model_id)
    model = VisionModelClass.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
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
    ann_file = "/root/harvested_dataset/dataset_annotations.json"
    img_dir = "/root/harvested_dataset/images"
    with open(ann_file, "r") as f:
        samples = json.load(f)
    print(f"Loaded {len(samples)} training samples from {ann_file}")

    class FastDataset(Dataset):
        def __init__(self, data):
            self.data = data
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            item = self.data[idx]
            img = Image.open(os.path.join(img_dir, item["image"])).convert("RGB")
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

    loader = DataLoader(FastDataset(samples), batch_size=1, shuffle=True, collate_fn=collate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    grad_accum_steps = 4

    for epoch in range(epochs):
        ep_loss = 0.0
        optimizer.zero_grad()
        for step, batch in enumerate(loader):
            b = {k: v.to(device) for k, v in batch.items()}
            out = model(**b)
            loss = out.loss / grad_accum_steps
            loss.backward()
            ep_loss += (loss.item() * grad_accum_steps)
            
            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(loader):
                optimizer.step()
                optimizer.zero_grad()
                torch.cuda.empty_cache()

        avg = ep_loss / len(loader)
        print(f"Epoch {epoch+1}/{epochs} | Average Training Loss: {avg:.4f}")
        # Save checkpoint after each epoch so intermediate progress is never lost
        ep_dir = f"/adapters/smolvlm_lora_{mode}"
        model.save_pretrained(ep_dir)
        processor.save_pretrained(ep_dir)
        volume.commit()

    out_path = f"/adapters/smolvlm_lora_{mode}"
    model.save_pretrained(out_path)
    processor.save_pretrained(out_path)
    volume.commit()
    print(f"Model saved and committed to Modal Volume: {out_path}")
    return f"Success: {out_path} (Final Loss: {avg:.4f})"


@app.local_entrypoint()
def main(mode: str = "both"):
    """
    Train SmolVLM LoRA adapters on Modal GPU.
    Usage:
      modal run modal_train_smolvlm.py --mode cot
      modal run modal_train_smolvlm.py --mode direct
      modal run modal_train_smolvlm.py --mode both
    """
    print("=================================================================")
    print(f"SMOLVLM LORA TRAINING ON MODAL GPU (TESLA T4) | MODE: {mode.upper()}")
    print("=================================================================")

    if mode in ["cot", "both"]:
        print("\n>>> Launching Short CoT Adapter Fine-Tuning (smolvlm_lora_cot)...")
        res_cot = train_smolvlm_modal.remote(mode="cot", epochs=3)
        print(f"Result Short CoT: {res_cot}")

    if mode in ["direct", "both"]:
        print("\n>>> Launching Direct Structured Tokens Adapter (smolvlm_lora_direct)...")
        res_dir = train_smolvlm_modal.remote(mode="direct", epochs=3)
        print(f"Result Direct Tokens: {res_dir}")

    print("\n=================================================================")
    print("ALL FINISHED! Adapters saved to Modal persistent volume:")
    print("  - /adapters/smolvlm_lora_cot")
    print("  - /adapters/smolvlm_lora_direct")
    print("=================================================================")
