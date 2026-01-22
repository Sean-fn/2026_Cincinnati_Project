#!/bin/bash
# Complete evaluation pipeline for KAN + QLoRA model
# Runs inference and evaluation, then compares with original M2F2-Det

set -e

source venv/bin/activate
current_path=$(pwd)
export PYTHONPATH="$current_path:$PYTHONPATH"

echo "=========================================="
echo "KAN + QLoRA Complete Evaluation Pipeline"
echo "=========================================="

# Step 1: Run detection inference
echo ""
echo "[1/4] Running detection inference..."
bash scripts/inference_kan_qlora_det.sh

# Rename output for comparison
mv outputs/DDVQA/DDVQA_det_c40.jsonl outputs/DDVQA/DDVQA_det_c40_kan.jsonl

# Step 2: Run explanation inference
echo ""
echo "[2/4] Running explanation inference..."
bash scripts/inference_kan_qlora_exp.sh

# Rename output for comparison
mv outputs/DDVQA/DDVQA_exp_c40.jsonl outputs/DDVQA/DDVQA_exp_c40_kan.jsonl

# Step 3: Evaluate detection performance
echo ""
echo "[3/4] Evaluating detection performance..."
echo "=========================================="
echo "KAN + QLoRA Detection Results:"
echo "=========================================="
python eval/eval_judgement.py --predict_path outputs/DDVQA/DDVQA_det_c40_kan.jsonl

# Step 4: Evaluate explanation performance
echo ""
echo "[4/4] Evaluating explanation performance..."
echo "=========================================="
echo "KAN + QLoRA Explanation Results:"
echo "=========================================="
python eval/eval_explanation.py --predict_path outputs/DDVQA/DDVQA_exp_c40_kan.jsonl

echo ""
echo "=========================================="
echo "Evaluation Complete!"
echo "=========================================="
echo ""
echo "To compare with original M2F2-Det, run:"
echo "  bash scripts/eval_original.sh"
echo ""
echo "Results saved to:"
echo "  - outputs/DDVQA/DDVQA_det_c40_kan.jsonl"
echo "  - outputs/DDVQA/DDVQA_exp_c40_kan.jsonl"
echo "=========================================="
