import os
import modal

print("[Direct Download] Connecting to Modal smolvlm-adapters-vol...")
vol = modal.Volume.from_name("smolvlm-adapters-vol")

targets = ["smolvlm_lora_cot", "smolvlm_lora_direct"]

for target in targets:
    out_dir = os.path.abspath(target)
    os.makedirs(out_dir, exist_ok=True)
    print(f"\nProcessing '{target}' into {out_dir}...")
    
    entries = list(vol.iterdir(target))
    print(f"Found {len(entries)} files in '{target}'.")
    for entry in entries:
        filename = os.path.basename(entry.path)
        out_path = os.path.join(out_dir, filename)
        print(f"  Reading {entry.path} ({entry.type})...")
        with open(out_path, "wb") as f:
            for chunk in vol.read_file(entry.path):
                f.write(chunk)
        size_kb = os.path.getsize(out_path) / 1024.0
        print(f"  Downloaded: {filename} ({size_kb:.1f} KB)")

print("\n[SUCCESS] All adapter files directly downloaded to disk!")
