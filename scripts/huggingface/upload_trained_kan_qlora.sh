#!/bin/bash
# 上传 stage_2_3 训练输出（仅 LoRA/KAN 权重）
# 用法: bash scripts/huggingface/upload_trained_kan_qlora.sh [your-hf-username]

set -e

# 配置（请修改为你的信息）
HF_USERNAME="${1:-YOUR_HF_USERNAME}"  # Hugging Face用户名
HF_REPO="M2F2-Det-Weights"            # 仓库名称
HF_REPO_FULL="${HF_USERNAME}/${HF_REPO}"

KAN_Q_LORA_DIR="checkpoints/llava-v1.5-7b-deepfake-kan-qlora"

echo "=========================================="
echo "上传训练输出 (KAN + QLoRA)"
echo "仓库: ${HF_REPO_FULL}"
echo "目录: ${KAN_Q_LORA_DIR}"
echo "=========================================="

# 检查是否安装huggingface_hub
if ! python -c "import huggingface_hub" 2>/dev/null; then
    echo "安装 huggingface_hub..."
    pip install huggingface_hub
fi

# 检查是否登录
echo -e "\n[1/3] 检查Hugging Face登录状态..."
if ! huggingface-cli whoami 2>/dev/null; then
    echo "请先登录Hugging Face:"
    echo "运行: huggingface-cli login"
    exit 1
fi

echo "✓ 已登录"

# 创建或连接到仓库
echo -e "\n[2/3] 创建/连接到HF仓库..."
python - <<EOF
from huggingface_hub import HfApi, create_repo

api = HfApi()
try:
    create_repo("${HF_REPO}", repo_type="model", exist_ok=True)
    print("✓ 仓库已准备: ${HF_REPO_FULL}")
except Exception as e:
    print(f"⚠️  仓库创建失败: {e}")
    print("请手动创建: https://huggingface.co/new")
    exit(1)
EOF

# 上传训练输出
echo -e "\n[3/3] 上传训练输出..."
if [ -d "${KAN_Q_LORA_DIR}" ]; then
    huggingface-cli upload \
        "${HF_REPO_FULL}" \
        "${KAN_Q_LORA_DIR}" \
        llava-v1.5-7b-deepfake-kan-qlora \
        --repo-type=model
    echo "✓ 训练输出上传完成"
else
    echo "⚠️  找不到 ${KAN_Q_LORA_DIR}"
    exit 1
fi

echo ""
echo "=========================================="
echo "✅ 上传完成！"
echo "=========================================="
echo ""
echo "Hugging Face仓库: https://huggingface.co/${HF_REPO_FULL}"
