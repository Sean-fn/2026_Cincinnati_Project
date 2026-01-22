#!/bin/bash
# Compare KAN + QLoRA results with Original M2F2-Det
# Prerequisites: Run eval_kan_qlora.sh and eval_original.sh first

source venv/bin/activate
current_path=$(pwd)
export PYTHONPATH="$current_path:$PYTHONPATH"

echo "=========================================="
echo "Performance Comparison Report"
echo "=========================================="
echo ""

# Check if result files exist
if [ ! -f "outputs/DDVQA/DDVQA_det_c40_kan.jsonl" ]; then
    echo "❌ KAN+QLoRA detection results not found!"
    echo "   Run: bash scripts/kan/eval.sh"
    exit 1
fi

if [ ! -f "outputs/DDVQA/DDVQA_det_c40_original.jsonl" ]; then
    echo "❌ Original detection results not found!"
    echo "   Run: bash scripts/eval/eval_original.sh"
    exit 1
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 DETECTION PERFORMANCE (Binary Classification)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "🔹 Original M2F2-Det:"
echo "────────────────────────────────────────"
python eval/eval_judgement.py --predict_path outputs/DDVQA/DDVQA_det_c40_original.jsonl
echo ""

echo "🔹 KAN + QLoRA:"
echo "────────────────────────────────────────"
python eval/eval_judgement.py --predict_path outputs/DDVQA/DDVQA_det_c40_kan.jsonl
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📝 EXPLANATION PERFORMANCE (NLG Metrics)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ -f "outputs/DDVQA/DDVQA_exp_c40_original.jsonl" ] && [ -f "outputs/DDVQA/DDVQA_exp_c40_kan.jsonl" ]; then
    echo "🔹 Original M2F2-Det:"
    echo "────────────────────────────────────────"
    python eval/eval_explanation.py --predict_path outputs/DDVQA/DDVQA_exp_c40_original.jsonl
    echo ""

    echo "🔹 KAN + QLoRA:"
    echo "────────────────────────────────────────"
    python eval/eval_explanation.py --predict_path outputs/DDVQA/DDVQA_exp_c40_kan.jsonl
    echo ""
else
    echo "⚠️  Explanation results not found. Skipping NLG evaluation."
    echo ""
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Comparison Complete"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📁 Result files:"
echo "  Original: outputs/DDVQA/DDVQA_{det,exp}_c40_original.jsonl"
echo "  KAN+QLo:  outputs/DDVQA/DDVQA_{det,exp}_c40_kan.jsonl"
echo ""
