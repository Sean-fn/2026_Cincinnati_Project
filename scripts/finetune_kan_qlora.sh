#!/bin/bash
# KAN + 4-bit QLoRA Training Script for M2F2-Det
# This script trains the Efficient KAN projector together with LoRA adapters
# to allow the LLM to adapt to the new KAN embedding distribution.
#
# Target GPU: RTX 4080 SUPER (16GB)
# Estimated VRAM: ~10-12 GB

set -e

# Activate environment
source /home/naeem/sean/M2F2_Det/venv/bin/activate
current_path=$(pwd)
export PYTHONPATH="$current_path:$PYTHONPATH"

# Configuration
CUDA_NUM=0
MODEL_VERSION="./checkpoints/llava-1.5-7b-deepfake-rand-proj-v1"
DATA_PATH="./utils/DDVQA_split/c40/train_DDVQA_format.json"
IMG_FOLDER="./utils/DDVQA_images/c40/train"
OUTPUT_DIR="./checkpoints/llava-v1.5-7b-deepfake-kan-qlora"
DEEPFAKE_CKPT_PATH="./utils/weights/M2F2_Det_densenet121.pth"
VISION_TOWER="openai/clip-vit-large-patch14-336"

# Create output directory
mkdir -p $OUTPUT_DIR

echo "========================================"
echo "M2F2-Det KAN + QLoRA Training"
echo "========================================"
echo "Model: $MODEL_VERSION"
echo "Output: $OUTPUT_DIR"
echo "GPU: $CUDA_NUM"
echo "========================================"

CUDA_VISIBLE_DEVICES=$CUDA_NUM python llava/train/train_deepfake.py \
    --model_name_or_path $MODEL_VERSION \
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
    --per_device_train_batch_size 6 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 20 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
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
echo "Training completed!"
echo "Output saved to: $OUTPUT_DIR"
echo "========================================"
