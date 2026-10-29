#!/usr/bin/env bash
# ==============================================================================
# ONESHOT_SETUP_AND_TRAIN.sh
# Complete zero-error harness for remote GPU VM (L40S / A100 / etc.)
# 1. Enforces numpy < 2.0.0 (prevents tensorstore segfault)
# 2. Enforces transformers < 5.0.0 (restores FlaxAutoModel / FlaxT5EncoderModel)
# 3. Downloads the official rail-berkeley/octo-small snapshot via huggingface-cli
# 4. Launches train_dobot_octo_head.py with GPU auto-tuning
# ==============================================================================

set -eo pipefail

BOLD='\033[1m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RESET='\033[0m'

echo -e "${BOLD}${CYAN}======================================================================${RESET}"
echo -e "${BOLD}${GREEN}        ONESHOT OCTO VLA GPU REPAIR & TRAINING LAUNCHER               ${RESET}"
echo -e "${BOLD}${CYAN}======================================================================${RESET}"

# Step 1: Install exact package versions that match Octo's Flax/JAX backend
echo -e "\n${CYAN}[1/4] Installing verified dependencies...${RESET}"
pip install "numpy<2.0.0" "transformers>=4.40.0,<5.0.0" tf-keras huggingface_hub

# Step 2: Download complete, valid checkpoint snapshot directly into octo_small_checkpoint/
echo -e "\n${CYAN}[2/4] Downloading full pretrained Octo Small checkpoint...${RESET}"
python -c "
import os
import shutil
import huggingface_hub

print('Fetching complete rail-berkeley/octo-small repository snapshot...')
snapshot_path = huggingface_hub.snapshot_download(
    repo_id='rail-berkeley/octo-small',
    local_dir='octo_small_checkpoint',
    local_dir_use_symlinks=False
)
print(f'Snapshot downloaded to: {snapshot_path}')

# Ensure 300000 step symlink exists for Orbax
src = os.path.join('octo_small_checkpoint', '270000')
dst = os.path.join('octo_small_checkpoint', '300000')
if os.path.exists(src) and not os.path.exists(dst):
    try:
        os.symlink('270000', dst)
        print('Created symlink: 300000 -> 270000')
    except Exception:
        shutil.copytree(src, dst)
        print('Copied tree: 270000 -> 300000')
"

# Step 3: Run Environment Verification
echo -e "\n${CYAN}[3/4] Running Environment Diagnostic Verification...${RESET}"
python test_vla_env.py

# Step 4: Execute High-Throughput Scaled Training
echo -e "\n${CYAN}[4/4] Launching GPU Training...${RESET}"
./autotrainscale.sh
