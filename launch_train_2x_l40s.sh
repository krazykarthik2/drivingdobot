#!/bin/bash
# ==============================================================================
# launch_train_2x_l40s.sh
# Multi-GPU Execution: Qwen2-VL-7B Teacher on GPU 1 + SmolVLM Student on GPU 0
# Target: 2x NVIDIA L40S (48GB VRAM each = 96GB Total)
# ==============================================================================

set -e

echo "=== System GPU Environment ==="
nvidia-smi

# Fix brotlicffi / httpx decompression compatibility issue
pip install --upgrade "brotli>=1.1.0" "brotlicffi>=1.1.0.0" httpx huggingface_hub
pip install --upgrade transformers accelerate peft qwen_vl_utils torchvision

export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export HF_HUB_ENABLE_HF_TRANSFER=0

# 1. Train Chain-of-Thought (CoT) with Qwen2-VL-7B Teacher & Live Inspection
echo ""
echo "=========================================================================="
echo ">>> STEP 1: Training Chain-of-Thought (CoT) LoRA with Qwen2-VL-7B Distillation"
echo "=========================================================================="
python train_distilled_lora_2x_l40s.py \
    --mode cot \
    --dataset_json harvested_dataset/dataset_annotations.json \
    --img_dir harvested_dataset/images \
    --output_dir smolvlm_lora_l40s_cot \
    --epochs 5 \
    --batch_size 16 \
    --grad_accum 2 \
    --lr 1.5e-4 \
    --lora_rank 32 \
    --use_teacher \
    --teacher_id Qwen/Qwen2-VL-7B-Instruct

# 2. Train Direct Structured Tokens LoRA
echo ""
echo "=========================================================================="
echo ">>> STEP 2: Training Direct Structured Tokens LoRA"
echo "=========================================================================="
python train_distilled_lora_2x_l40s.py \
    --mode direct \
    --dataset_json harvested_dataset/dataset_annotations.json \
    --img_dir harvested_dataset/images \
    --output_dir smolvlm_lora_l40s_direct \
    --epochs 5 \
    --batch_size 16 \
    --grad_accum 2 \
    --lr 1.5e-4 \
    --lora_rank 32

echo ""
echo "=========================================================================="
echo ">>> ALL L40S TRAININGS COMPLETED!"
echo ">>> Output Adapters:"
echo "    - CoT: smolvlm_lora_l40s_cot"
echo "    - Direct: smolvlm_lora_l40s_direct"
echo "=========================================================================="
