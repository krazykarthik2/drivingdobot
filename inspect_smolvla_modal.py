import modal

app = modal.App("inspect-smolvla")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "lerobot",
        "peft>=0.14.0",
        "torchvision",
        "num2words"
    )
)

@app.function(image=image, gpu="T4", timeout=300)
def inspect():
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    print("Loading SmolVLAPolicy from lerobot/smolvla_base...")
    policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
    print("\n[SmolVLA Policy Class]:", type(policy))
    print("[SmolVLA Model Class]:", type(policy.model))
    
    print("\n--- Children of policy.model ---")
    for name, child in policy.model.named_children():
        total_p = sum(p.numel() for p in child.parameters())
        print(f"  {name}: {type(child).__name__} ({total_p/1e6:.2f}M params)")

    print("\n--- Model Config ---")
    for k, v in policy.config.__dict__.items():
        if not k.startswith("_"):
            print(f"  {k}: {v}")

@app.local_entrypoint()
def main():
    inspect.remote()
