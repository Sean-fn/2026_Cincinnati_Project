#!/bin/bash
# 从 Hugging Face Hub 下载大文件
# 用法: bash scripts/download_from_huggingface.sh [hf-username]

set -e

# 配置
HF_USERNAME="${1:-YOUR_HF_USERNAME}"
HF_REPO="M2F2-Det-Weights"
HF_REPO_FULL="${HF_USERNAME}/${HF_REPO}"

echo "=========================================="
echo "从Hugging Face下载M2F2_Det权重"
echo "仓库: ${HF_REPO_FULL}"
echo "=========================================="

# 检查Python和pip
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到python3"
    exit 1
fi

# 安装huggingface_hub
echo -e "\n[1/4] 安装依赖..."
pip install -q huggingface_hub

# 创建目录
echo -e "\n[2/4] 创建目录..."
mkdir -p checkpoints
mkdir -p utils/weights

# 下载模型
echo -e "\n[3/4] 下载大文件..."
echo "这可能需要一些时间（总计~16GB）..."

python - <<EOF
from huggingface_hub import snapshot_download, hf_hub_download
import os

repo_id = "${HF_REPO_FULL}"

print("\n下载 llava-1.5-7b-deepfake-rand-proj-v1 (14GB)...")
try:
    snapshot_download(
        repo_id=repo_id,
        allow_patterns="llava-1.5-7b-deepfake-rand-proj-v1/*",
        local_dir="./checkpoints",
        local_dir_use_symlinks=False
    )
    print("✓ 模型下载完成")
except Exception as e:
    print(f"❌ 模型下载失败: {e}")
    print(f"请手动下载: https://huggingface.co/{repo_id}")

print("\n下载 M2F2_Det_densenet121.pth (1.7GB)...")
try:
    hf_hub_download(
        repo_id=repo_id,
        filename="weights/M2F2_Det_densenet121.pth",
        local_dir="./utils",
        local_dir_use_symlinks=False
    )
    print("✓ 权重下载完成")
except Exception as e:
    print(f"❌ 权重下载失败: {e}")
    print(f"请手动下载: https://huggingface.co/{repo_id}")

EOF

# 验证下载
echo -e "\n[4/4] 验证下载..."

if [ -d "checkpoints/llava-1.5-7b-deepfake-rand-proj-v1" ]; then
    echo "✓ checkpoints/llava-1.5-7b-deepfake-rand-proj-v1"
    ls -lh checkpoints/llava-1.5-7b-deepfake-rand-proj-v1/*.safetensors | head -3
else
    echo "❌ 缺失: checkpoints/llava-1.5-7b-deepfake-rand-proj-v1"
fi

if [ -f "utils/weights/M2F2_Det_densenet121.pth" ]; then
    echo "✓ utils/weights/M2F2_Det_densenet121.pth"
    ls -lh utils/weights/M2F2_Det_densenet121.pth
else
    echo "❌ 缺失: utils/weights/M2F2_Det_densenet121.pth"
fi

echo ""
echo "=========================================="
echo "✅ 下载完成！"
echo "=========================================="
echo ""
echo "文件位置:"
echo "  - checkpoints/llava-1.5-7b-deepfake-rand-proj-v1/"
echo "  - utils/weights/M2F2_Det_densenet121.pth"
echo ""
echo "下一步: bash scripts/verify_local.sh"
