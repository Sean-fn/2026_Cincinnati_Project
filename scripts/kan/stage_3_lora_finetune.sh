#!/bin/bash
# Stage 3: LoRA Fine-tuning with KAN Projector
# This script fine-tunes the LLM using LoRA while also fine-tuning the KAN projector
# trained in Stage 2. Uses 4-bit quantization for memory efficiency.
#
# Target GPU: RTX 4080 SUPER (16GB)
# Estimated VRAM: ~12-14 GB

set -e

# Activate environment
current_path=$(pwd)
export PYTHONPATH="$current_path:$PYTHONPATH"

# Configuration
CUDA_NUM=0
# Use the Stage 2 checkpoint as starting point
STAGE2_MODEL="./checkpoints/llava-v1.5-7b-deepfake-kan-stage2"
DATA_PATH="./utils/DDVQA_split/c40/train_DDVQA_format.json"
IMG_FOLDER="./utils/DDVQA_images/c40/train"
OUTPUT_DIR="./checkpoints/llava-v1.5-7b-deepfake-kan-lora-stage3"
DEEPFAKE_CKPT_PATH="./utils/weights/M2F2_Det_densenet121.pth"
VISION_TOWER="openai/clip-vit-large-patch14-336"

# Create output directory
mkdir -p $OUTPUT_DIR

echo "========================================"
echo "Stage 3: LoRA Fine-tuning"
echo "========================================"
echo "Base Model: $STAGE2_MODEL"
echo "Output: $OUTPUT_DIR"
echo "GPU: $CUDA_NUM"
echo "Training: LLM (LoRA) + KAN Projector"
echo "Quantization: 4-bit NF4"
echo "========================================"

CUDA_VISIBLE_DEVICES=$CUDA_NUM python llava/train/train_deepfake.py \
    --model_name_or_path $STAGE2_MODEL \
    --version v1 \
    --data_path $DATA_PATH \
    --image_folder $IMG_FOLDER \
    --vision_tower $VISION_TOWER \
    --deepfake_ckpt_path $DEEPFAKE_CKPT_PATH \
    --lora_enable True \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --bits 4 \
    --double_quant True \
    --quant_type nf4 \
    --deepfake_projector_type efficient_kan \
    --kan_hidden_dim 128 \
    --kan_grid_size 5 \
    --kan_spline_order 3 \
    --tune_mm_mlp_adapter True \
    --tune_deepfake_mlp_adapter True \
    --freeze_backbone False \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_vision_select_feature cls_patch \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --mm_projector_lr 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True

echo "========================================"
echo "Stage 3 Training Completed!"
echo "Output saved to: $OUTPUT_DIR"
echo ""
echo "Model is ready for inference!"
echo "Use scripts/kan/inference_*.sh for testing"
echo "========================================"
