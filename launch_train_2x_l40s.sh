#!/bin/bash
# ==============================================================================
# launch_train_2x_l40s.sh
# Multi-GPU Execution Launcher for 2x NVIDIA L40S (48GB each)
# ==============================================================================

set -e

echo "=== System Environment Check ==="
nvidia-smi

export OMP_NUM_THREADS=8
export NCCL_DEBUG=INFO
export PYTHONUNBUFFERED=1

# 1. Train Chain-of-Thought (CoT) LoRA with Teacher Distillation
echo ""
echo "=========================================================================="
echo ">>> STEP 1: Training Chain-of-Thought (CoT) LoRA with 2x L40S Distillation"
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
    --use_distillation \
    --teacher_id HuggingFaceTB/SmolVLM2-500M-Video-Instruct

# 2. Train Direct Structured Tokens LoRA
echo ""
echo "=========================================================================="
echo ">>> STEP 2: Training Direct Structured Tokens LoRA with 2x L40S"
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
echo ">>> ALL L40S TRAININGS COMPLETED SUCCESSFULLY!"
echo ">>> Generated Adapters:"
echo "    - CoT: smolvlm_lora_l40s_cot"
echo "    - Direct: smolvlm_lora_l40s_direct"
echo "=========================================================================="
