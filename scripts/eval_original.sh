#!/bin/bash
# Evaluation pipeline for original M2F2-Det model
# For comparison with KAN + QLoRA approach

set -e

source venv/bin/activate
current_path=$(pwd)
export PYTHONPATH="$current_path:$PYTHONPATH"

echo "=========================================="
echo "Original M2F2-Det Evaluation Pipeline"
echo "=========================================="

# Step 1: Run detection inference
echo ""
echo "[1/4] Running detection inference..."
bash stage_3_inference_det.sh

# Backup original output
cp outputs/DDVQA/DDVQA_det_c40.jsonl outputs/DDVQA/DDVQA_det_c40_original.jsonl

# Step 2: Run explanation inference
echo ""
echo "[2/4] Running explanation inference..."
bash stage_3_inference_exp.sh

# Backup original output
cp outputs/DDVQA/DDVQA_exp_c40.jsonl outputs/DDVQA/DDVQA_exp_c40_original.jsonl

# Step 3: Evaluate detection performance
echo ""
echo "[3/4] Evaluating detection performance..."
echo "=========================================="
echo "Original M2F2-Det Detection Results:"
echo "=========================================="
python eval/eval_judgement.py --predict_path outputs/DDVQA/DDVQA_det_c40_original.jsonl

# Step 4: Evaluate explanation performance
echo ""
echo "[4/4] Evaluating explanation performance..."
echo "=========================================="
echo "Original M2F2-Det Explanation Results:"
echo "=========================================="
python eval/eval_explanation.py --predict_path outputs/DDVQA/DDVQA_exp_c40_original.jsonl

echo ""
echo "=========================================="
echo "Evaluation Complete!"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  - outputs/DDVQA/DDVQA_det_c40_original.jsonl"
echo "  - outputs/DDVQA/DDVQA_exp_c40_original.jsonl"
echo "=========================================="
