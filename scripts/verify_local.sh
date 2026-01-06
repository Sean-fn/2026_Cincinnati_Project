#!/bin/bash
# 本地快速验证脚本 - 确保代码可以在远程运行
# 用途：验证代码逻辑、依赖、配置，但不运行完整训练

set -e  # 遇到错误立即退出

echo "=========================================="
echo "M2F2_Det 本地验证脚本"
echo "=========================================="

# 1. 检查环境
echo -e "\n[1/6] 检查Python环境..."
python --version
pip list | grep -E "torch|transformers|deepspeed|llava"

# 2. 检查CUDA（如果有GPU）
echo -e "\n[2/6] 检查CUDA..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv
    python -c "import torch; print(f'PyTorch CUDA: {torch.cuda.is_available()}')"
else
    echo "⚠️  未检测到GPU（本地开发可忽略）"
fi

# 3. 检查关键文件
echo -e "\n[3/6] 检查关键文件..."
FILES=(
    "llava/model/deepfake/M2F2Det/model.py"
    "llava/train/train_deepfake.py"
    "scripts/finetune_stage_2.sh"
    "checkpoints/llava-v1.5-7b/config.json"
)

for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "✓ $file"
    else
        echo "✗ 缺失: $file"
        exit 1
    fi
done

# 4. 验证导入（不加载模型权重）
echo -e "\n[4/6] 验证Python导入..."
python -c "
import sys
sys.path.insert(0, '.')

# 测试导入
from llava.model.deepfake.M2F2Det.model import M2F2Det
from llava.train.train_deepfake import train
from llava.model.builder import load_pretrained_model
print('✓ 所有模块导入成功')
"

# 5. Dry-run训练脚本（只运行1个step）
echo -e "\n[5/6] Dry-run训练脚本..."
python llava/train/train_deepfake.py \
    --model_name_or_path ./checkpoints/llava-v1.5-7b \
    --data_path ./utils/DDVQA_split/c40/train_DDVQA_format_judge_only.json \
    --image_folder ./utils/DDVQA_images/c40/train \
    --output_dir ./test_output_dry_run \
    --max_steps 1 \
    --per_device_train_batch_size 1 \
    --logging_steps 1 \
    --save_strategy "no" \
    --evaluation_strategy "no" \
    --report_to "none" \
    --bf16 True \
    --deepfake_ckpt_path ./utils/weights/M2F2_Det_densenet121.pth \
    --tune_deepfake_mlp_adapter True \
    --freeze_mm_mlp_adapter True \
    --freeze_backbone True \
    --mm_vision_select_feature cls_patch \
    || { echo "❌ Dry-run失败！检查配置"; exit 1; }

echo "✓ Dry-run成功"

# 6. 清理测试输出
echo -e "\n[6/6] 清理测试文件..."
rm -rf ./test_output_dry_run

echo -e "\n=========================================="
echo "✅ 验证通过！代码可以部署到远程服务器"
echo "=========================================="
echo ""
echo "下一步："
echo "  1. 提交代码: git add . && git commit -m 'verified locally'"
echo "  2. 同步到远程: bash scripts/sync_to_remote.sh"
echo "  3. 远程训练: ssh remote 'cd M2F2_Det && bash stage_2_train.sh'"
