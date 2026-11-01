import os
import modal

app = modal.App("smolvlm-downloader")
volume = modal.Volume.from_name("smolvlm-adapters-vol")

@app.function(volumes={"/adapters": volume}, timeout=300)
def read_adapter_file(adapter_name: str, filename: str) -> bytes:
    path = f"/adapters/{adapter_name}/{filename}"
    with open(path, "rb") as f:
        return f.read()

@app.local_entrypoint()
def main(adapter_name: str = "smolvlm_lora_cot"):
    out_dir = os.path.abspath(adapter_name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Downloading '{adapter_name}' from Modal volume into {out_dir}...")

    files = [
        "adapter_config.json",
        "adapter_model.safetensors",
        "chat_template.jinja",
        "processor_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "README.md"
    ]

    for fn in files:
        data = read_adapter_file.remote(adapter_name, fn)
        out_f = os.path.join(out_dir, fn)
        with open(out_f, "wb") as f:
            f.write(data)
        print(f"  ✓ Downloaded {fn} ({len(data) / 1024:.1f} KB)")

    print(f"\n[SUCCESS]: '{adapter_name}' downloaded successfully!")
