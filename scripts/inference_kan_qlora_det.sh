#!/bin/bash
# KAN + QLoRA Detection Inference
# Evaluates binary classification (real/fake) on DDVQA dataset

source venv/bin/activate
current_path=$(pwd)
export PYTHONPATH="$current_path:$PYTHONPATH"

CUDA_NUM=0
MODEL_PATH="./checkpoints/llava-v1.5-7b-deepfake-kan-qlora"

echo "========================================"
echo "KAN + QLoRA Detection Inference"
echo "========================================"
echo "Model: $MODEL_PATH"
echo "GPU: $CUDA_NUM"
echo "========================================"

CUDA_VISIBLE_DEVICES=$CUDA_NUM python -m llava.serve.cli_DDVQA_det \
    --model-path $MODEL_PATH \
    --load-4bit

echo "========================================"
echo "Detection inference completed!"
echo "Output: outputs/DDVQA/DDVQA_det_c40.jsonl"
echo "========================================"
