#!/bin/bash
# Stage 2: KAN Projector Training Only
# This script trains ONLY the KAN-based deepfake projector while freezing the LLM.
# The KAN projector learns to map binary detection outputs to LLM hidden space.
#
# Estimated VRAM: ~10-12 GB

set -e

# Activate environment
current_path=$(pwd)
export PYTHONPATH="$current_path:$PYTHONPATH"

# Configuration
CUDA_NUM=0
BASE_MODEL="./checkpoints/llava-v1.5-7b-deepfake-rand-proj-v1"
DATA_PATH="./utils/DDVQA_split/c40/train_DDVQA_format.json"
IMG_FOLDER="./utils/DDVQA_images/c40/train"
OUTPUT_DIR="./checkpoints/llava-v1.5-7b-deepfake-kan-stage2"
DEEPFAKE_CKPT_PATH="./utils/weights/M2F2_Det_densenet121.pth"
VISION_TOWER="openai/clip-vit-large-patch14-336"

# Create output directory
mkdir -p $OUTPUT_DIR

echo "========================================"
echo "Stage 2: KAN Projector Training"
echo "========================================"
echo "Base Model: $BASE_MODEL"
echo "Output: $OUTPUT_DIR"
echo "GPU: $CUDA_NUM"
echo "Training: KAN Projector ONLY"
echo "Frozen: LLM, Vision Tower, MM Projector"
echo "========================================"

CUDA_VISIBLE_DEVICES=$CUDA_NUM python llava/train/train_deepfake.py \
    --model_name_or_path $BASE_MODEL \
    --version v1 \
    --data_path $DATA_PATH \
    --image_folder $IMG_FOLDER \
    --vision_tower $VISION_TOWER \
    --deepfake_ckpt_path $DEEPFAKE_CKPT_PATH \
    --lora_enable False \
    --deepfake_projector_type efficient_kan \
    --kan_hidden_dim 128 \
    --kan_grid_size 5 \
    --kan_spline_order 3 \
    --tune_mm_mlp_adapter False \
    --tune_deepfake_mlp_adapter True \
    --freeze_backbone True \
    --freeze_mm_mlp_adapter True \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_vision_select_feature cls_patch \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 3 \
    --learning_rate 5e-4 \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none

echo "========================================"
echo "Stage 2 Training Completed!"
echo "Output saved to: $OUTPUT_DIR"
echo ""
echo "Next Step:"
echo "Run Stage 3 LoRA fine-tuning:"
echo "  bash scripts/kan/stage_3_lora_finetune.sh"
echo "========================================"
