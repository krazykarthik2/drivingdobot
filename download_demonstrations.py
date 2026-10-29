import os
import sys
import urllib.request

DATASET_NAME = "dobot_front_demonstrations.pkl"
HF_URL = f"https://huggingface.co/datasets/karthikkrazy/dobot-front-demonstrations/resolve/main/{DATASET_NAME}"

def download_dataset():
    if os.path.exists(DATASET_NAME) and os.path.getsize(DATASET_NAME) > 1000000:
        print(f"Dataset {DATASET_NAME} already exists ({os.path.getsize(DATASET_NAME) / 1024 / 1024:.1f} MB).")
        return

    print("=" * 65)
    print("DOWNLOADING DOBOT DEMONSTRATION DATASET FROM HUGGINGFACE")
    print(f"Target: {DATASET_NAME}")
    print("=" * 65)

    req = urllib.request.Request(HF_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, open(DATASET_NAME, "wb") as out_file:
        total_size = int(resp.headers.get("content-length", 0))
        downloaded = 0
        chunk_size = 1024 * 1024

        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                pct = int(downloaded * 100 / total_size)
                mb_d = downloaded / (1024 * 1024)
                mb_t = total_size / (1024 * 1024)
                sys.stdout.write(f"\r  Progress: {pct:02d}% ({mb_d:.1f}/{mb_t:.1f} MB)")
                sys.stdout.flush()

    print(f"\nSuccessfully downloaded {DATASET_NAME} ({os.path.getsize(DATASET_NAME) / 1024 / 1024:.1f} MB)!")

if __name__ == "__main__":
    download_dataset()
