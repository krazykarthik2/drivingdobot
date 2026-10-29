import os
import sys
import glob
import pickle
from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "karthikkrazy/dobot-front-demonstrations"
OUTPUT_FILE = "dobot_front_demonstrations.pkl"

def download_and_assemble(repo_id=REPO_ID, output_path=OUTPUT_FILE):
    print("=" * 70)
    print("DOWNLOADING DEMONSTRATION CHUNKS FROM HUGGINGFACE")
    print(f"Dataset Repository: {repo_id}")
    print("=" * 70)

    api = HfApi()
    repo_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    chunk_files = sorted([f for f in repo_files if f.startswith("chunks/") and f.endswith(".pkl")])

    if not chunk_files:
        print("No chunk files found in chunks/. Checking root directory...")
        chunk_files = sorted([f for f in repo_files if f.endswith(".pkl")])

    print(f"Found {len(chunk_files)} chunk files to assemble.")

    all_demonstrations = []
    for idx, cf in enumerate(chunk_files):
        print(f"[{idx+1}/{len(chunk_files)}] Downloading {cf}...")
        local_path = hf_hub_download(repo_id=repo_id, filename=cf, repo_type="dataset")
        with open(local_path, "rb") as f:
            chunk_data = pickle.load(f)
            all_demonstrations.extend(chunk_data)
        print(f"   Accumulated: {len(all_demonstrations)} total episodes so far.")

    print(f"\nWriting unified dataset to {output_path} ({len(all_demonstrations)} total episodes)...")
    with open(output_path, "wb") as f:
        pickle.dump(all_demonstrations, f)

    sz_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Successfully assembled {output_path} ({sz_mb:.2f} MB, {len(all_demonstrations)} episodes)!")

if __name__ == "__main__":
    download_and_assemble()
