#!/usr/bin/env bash
# ==============================================================================
# autotrainscale.sh - Dynamic Auto-Scaling Training Harness for Octo VLA
# Tuned for NVIDIA L40S (48GB Ada Lovelace VRAM), A100 (40GB/80GB), H100, RTX 4090
# Dynamically configures VRAM allocation, JAX memory preallocation, batch size,
# thread pooling, and system resources.
# ==============================================================================

set -eo pipefail

BOLD='\033[1m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
RESET='\033[0m'

echo -e "${BOLD}${BLUE}======================================================================${RESET}"
echo -e "${BOLD}${CYAN}        OCTO VLA AUTOTRAINSCALE: GPU & RESOURCE AUTO-TUNER            ${RESET}"
echo -e "${BOLD}${BLUE}======================================================================${RESET}"

# 0. Automatic Dependency Check & Auto-Installation
echo -e "${CYAN}[0/5] Checking Required Packages (Octo, dlimp, tf_keras, pycollada, xvfb, NumPy < 2.0, Transformers < 5.0)...${RESET}"
# Check if Xvfb or virtual display tools are needed for headless pyrender
if [ -z "$DISPLAY" ] && ! command -v Xvfb &> /dev/null; then
    echo -e "${YELLOW}Headless environment detected without Xvfb. Attempting to install xvfb...${RESET}"
    if command -v sudo &> /dev/null; then
        sudo apt-get update -qq && sudo apt-get install -y -qq xvfb libgl1-mesa-glx 2>/dev/null || true
    fi
fi

# Enforce pycollada for trimesh DAE CAD loading
python -c "import collada" 2>/dev/null || {
    echo -e "${YELLOW}pycollada missing for CAD mesh loading. Installing pycollada...${RESET}"
    pip install "pycollada>=0.8"
}

# Enforce transformers < 5.0.0 (Transformers v5 completely removed Flax/JAX support)
python -c "import transformers; assert int(transformers.__version__.split('.')[0]) < 5, 'Transformers v5 dropped Flax'" 2>/dev/null || {
    echo -e "${YELLOW}Transformers v5 detected. Downgrading to transformers==4.46.3...${RESET}"
    pip install "transformers>=4.40.0,<5.0.0"
}

# Enforce numpy < 2.0.0 for tensorstore and JAX C-extension ABI stability
python -c "import numpy as np; assert int(np.__version__.split('.')[0]) < 2, 'NumPy 2.x ABI mismatch'" 2>/dev/null || {
    echo -e "${YELLOW}NumPy 2.x detected or incompatible ABI. Downgrading to NumPy 1.26.4...${RESET}"
    pip install "numpy<2.0.0"
}

python -c "import octo; import dlimp; import tf_keras; import jax" 2>/dev/null || {
    echo -e "${YELLOW}Missing packages detected. Installing requirements...${RESET}"
    pip install "numpy<2.0.0" "transformers>=4.40.0,<5.0.0" tf-keras
    pip install -r requirements.txt
}

# Check if pretrained Octo Small weights exist locally, download if missing
if [ ! -f "octo_small_checkpoint/config.json" ] || [ ! -f "octo_small_checkpoint/270000/default/checkpoint" ]; then
    echo -e "${CYAN}Pretrained octo_small_checkpoint missing or incomplete. Downloading files directly...${RESET}"
    python download_octo_checkpoint.py
fi

# Clean up any invalid 300000 directory so Orbax strictly uses the canonical 270000 step
rm -rf octo_small_checkpoint/300000 2>/dev/null || true

# Check if demonstration dataset exists locally, download from HF if missing
if [ ! -f "dobot_front_demonstrations.pkl" ] || [ $(stat -c%s "dobot_front_demonstrations.pkl" 2>/dev/null || echo 0) -lt 1000000 ]; then
    echo -e "${CYAN}dobot_front_demonstrations.pkl missing or incomplete.${RESET}"
    echo -e "${CYAN}Downloading demonstration chunks from HuggingFace dataset (karthikkrazy/dobot-front-demonstrations)...${RESET}"
    python download_and_assemble_demos.py
fi
fi

# 1. Inspect System Architecture & Disk Space
AVAILABLE_DISK_GB=$(df -BG . | awk 'NR==2 {gsub(/G/,""); print $4}')
echo -e "${CYAN}[1/5] Checking Disk & Storage:${RESET} Available Workspace Disk: ${BOLD}${AVAILABLE_DISK_GB} GB${RESET}"
if [ "$AVAILABLE_DISK_GB" -lt 5 ]; then
    echo -e "${YELLOW}Warning: Low disk space (<5GB). Checkpoint saving might fail if space is exhausted.${RESET}"
fi

# 2. Inspect GPU & VRAM via nvidia-smi and JAX device detection
HAS_GPU=false
TOTAL_VRAM_MB=0
GPU_NAME="Unknown"

if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1 || echo "Unknown")
    TOTAL_VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1 || echo 0)
    TOTAL_VRAM_MB=$(echo "$TOTAL_VRAM_MB" | tr -d ' ')
fi

TOTAL_VRAM_GB=$((TOTAL_VRAM_MB / 1024))

echo -e "${CYAN}[2/5] Inspecting GPU Hardware & JAX CUDA Availability:${RESET}"
# Check if JAX can actually allocate and run on GPU without being locked or missing jaxlib-cuda
JAX_HAS_GPU=$(python -c "
import jax
try:
    devices = jax.devices('gpu')
    print('true' if len(devices) > 0 else 'false')
except Exception:
    print('false')
" 2>/dev/null || echo "false")

if [ "$JAX_HAS_GPU" = "true" ] && [ "$TOTAL_VRAM_MB" -gt 0 ]; then
    HAS_GPU=true
    echo -e "      Found GPU: ${BOLD}${GREEN}${GPU_NAME}${RESET}"
    echo -e "      Dedicated VRAM: ${BOLD}${GREEN}${TOTAL_VRAM_GB} GB (${TOTAL_VRAM_MB} MB)${RESET}"
    echo -e "      JAX CUDA Status: ${BOLD}${GREEN}ACCESSIBLE & UNLOCKED${RESET}"
else
    if [ "$TOTAL_VRAM_MB" -gt 0 ]; then
        echo -e "      Hardware GPU detected (${GPU_NAME}), but ${YELLOW}CUDA is locked, unavailable, or missing CUDA-enabled jaxlib${RESET}."
        echo -e "      ${YELLOW}==> Safely auto-falling back to CPU execution profile!${RESET}"
    else
        echo -e "      ${YELLOW}No active NVIDIA GPU detected. Using CPU execution profile.${RESET}"
    fi
    HAS_GPU=false
    TOTAL_VRAM_GB=0
fi

# 3. Dynamic Scaling Decision Engine based on VRAM & Availability
BATCH_SIZE=8
EPOCHS=30
LEARNING_RATE="2.5e-4"
JAX_MEM_PREALLOC="0.50"

if [ "$HAS_GPU" = true ] && [ "$TOTAL_VRAM_GB" -ge 40 ]; then
    echo -e "      ${BOLD}${GREEN}High-Tier Workstation / Datacenter Profile Active (L40S / A100 / H100)!${RESET}"
    BATCH_SIZE=64
    EPOCHS=40
    LEARNING_RATE="5e-4"
    JAX_MEM_PREALLOC="0.85"
elif [ "$HAS_GPU" = true ] && [ "$TOTAL_VRAM_GB" -ge 22 ]; then
    echo -e "      ${BOLD}${GREEN}Prosumer High-VRAM Profile Active (RTX 3090 / 4090 / 24GB)!${RESET}"
    BATCH_SIZE=32
    EPOCHS=35
    LEARNING_RATE="3.5e-4"
    JAX_MEM_PREALLOC="0.80"
elif [ "$HAS_GPU" = true ] && [ "$TOTAL_VRAM_GB" -ge 14 ]; then
    echo -e "      ${BOLD}${GREEN}Mid-VRAM Profile Active (T4 / V100 / 16GB)!${RESET}"
    BATCH_SIZE=16
    EPOCHS=30
    LEARNING_RATE="3e-4"
    JAX_MEM_PREALLOC="0.75"
else
    echo -e "      ${YELLOW}CPU-Safe Profile Active (Batch: 8, Auto-Tuned Threads).${RESET}"
    BATCH_SIZE=8
    EPOCHS=20
    LEARNING_RATE="2e-4"
    JAX_MEM_PREALLOC="0.30"
    export JAX_PLATFORMS=cpu
fi

# Allow CLI overrides if arguments provided: e.g. ./autotrainscale.sh --batch_size 128
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --batch_size) BATCH_SIZE="$2"; shift ;;
        --epochs) EPOCHS="$2"; shift ;;
        --lr) LEARNING_RATE="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

echo -e "${CYAN}[3/5] Auto-Scaled Configuration Parameters:${RESET}"
echo -e "      • Batch Size:       ${BOLD}${GREEN}${BATCH_SIZE}${RESET}"
echo -e "      • Training Epochs:  ${BOLD}${GREEN}${EPOCHS}${RESET}"
echo -e "      • Learning Rate:    ${BOLD}${GREEN}${LEARNING_RATE}${RESET}"
echo -e "      • XLA/JAX Prealloc: ${BOLD}${GREEN}${JAX_MEM_PREALLOC}${RESET}"

# 4. Environment Variables for JAX / XLA & PyTorch CUDA
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=${JAX_MEM_PREALLOC}
export TF_FORCE_GPU_ALLOW_GROWTH=true
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=$(nproc || echo 4)

# 5. Dependency & GPU Driver Verification
echo -e "${CYAN}[4/5] Checking JAX Accelerator Detection:${RESET}"
python -c "
import jax
devices = jax.devices()
print(f'      Active Backend:    {jax.default_backend().upper()}')
print(f'      Available Devices: {devices}')
"

# 6. Execute Scaled Training Run
echo -e "${CYAN}[5/5] Launching High-Throughput Octo Action Head Training...${RESET}"
echo -e "${BOLD}${BLUE}======================================================================${RESET}"

CMD="python train_dobot_octo_head.py \
    --dataset dobot_front_demonstrations.pkl \
    --checkpoint_dir octo_dobot_front_checkpoint \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --lr ${LEARNING_RATE}"

echo -e "Executing: ${CYAN}${CMD}${RESET}\n"
exec $CMD
