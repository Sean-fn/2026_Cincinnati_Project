#!/bin/bash
# 上传大文件到 Hugging Face Hub
# 用法: bash scripts/upload_to_huggingface.sh [your-hf-username]

set -e

# 配置（请修改为你的信息）
HF_USERNAME="${1:-YOUR_HF_USERNAME}"  # Hugging Face用户名
HF_REPO="M2F2-Det-Weights"            # 仓库名称
HF_REPO_FULL="${HF_USERNAME}/${HF_REPO}"

echo "=========================================="
echo "上传M2F2_Det权重到Hugging Face"
echo "仓库: ${HF_REPO_FULL}"
echo "=========================================="

# 检查是否安装huggingface_hub
if ! python -c "import huggingface_hub" 2>/dev/null; then
    echo "安装 huggingface_hub..."
    pip install huggingface_hub
fi

# 检查是否登录
echo -e "\n[1/4] 检查Hugging Face登录状态..."
if ! huggingface-cli whoami 2>/dev/null; then
    echo "请先登录Hugging Face:"
    echo "运行: huggingface-cli login"
    exit 1
fi

echo "✓ 已登录"

# 创建或连接到仓库
echo -e "\n[2/4] 创建/连接到HF仓库..."
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

# 上传文件
echo -e "\n[3/4] 上传大文件..."

# 上传 llava-1.5-7b-deepfake-rand-proj-v1
if [ -d "checkpoints/llava-1.5-7b-deepfake-rand-proj-v1" ]; then
    echo "上传 llava-1.5-7b-deepfake-rand-proj-v1 (14GB)..."
    huggingface-cli upload \
        "${HF_REPO_FULL}" \
        checkpoints/llava-1.5-7b-deepfake-rand-proj-v1 \
        llava-1.5-7b-deepfake-rand-proj-v1 \
        --repo-type=model
    echo "✓ llava-1.5-7b-deepfake-rand-proj-v1 上传完成"
else
    echo "⚠️  找不到 checkpoints/llava-1.5-7b-deepfake-rand-proj-v1"
fi

# 上传 M2F2_Det_densenet121.pth
if [ -f "utils/weights/M2F2_Det_densenet121.pth" ]; then
    echo "上传 M2F2_Det_densenet121.pth (1.7GB)..."
    huggingface-cli upload \
        "${HF_REPO_FULL}" \
        utils/weights/M2F2_Det_densenet121.pth \
        weights/M2F2_Det_densenet121.pth \
        --repo-type=model
    echo "✓ M2F2_Det_densenet121.pth 上传完成"
else
    echo "⚠️  找不到 utils/weights/M2F2_Det_densenet121.pth"
fi

# 创建 README
echo -e "\n[4/4] 创建README..."
cat > /tmp/hf_readme.md <<'EOFREADME'
# M2F2-Det Model Weights

Large files for the M2F2-Det project.

## Files

- `llava-1.5-7b-deepfake-rand-proj-v1/`: LLaVA model with deepfake detection initialization (14GB)
- `weights/M2F2_Det_densenet121.pth`: Stage-1 trained deepfake detector (1.7GB)

## Usage

Download these files to your project:

```bash
# Install huggingface_hub
pip install huggingface_hub

# Download script
bash scripts/download_from_huggingface.sh
```

Or manually:

```python
from huggingface_hub import hf_hub_download

# Download model
hf_hub_download(
    repo_id="YOUR_USERNAME/M2F2-Det-Weights",
    filename="llava-1.5-7b-deepfake-rand-proj-v1/model-00001-of-00003.safetensors",
    local_dir="./checkpoints"
)
```

## Source

GitHub: https://github.com/YOUR_USERNAME/M2F2_Det
EOFREADME

huggingface-cli upload \
    "${HF_REPO_FULL}" \
    /tmp/hf_readme.md \
    README.md \
    --repo-type=model

echo ""
echo "=========================================="
echo "✅ 上传完成！"
echo "=========================================="
echo ""
echo "Hugging Face仓库: https://huggingface.co/${HF_REPO_FULL}"
echo ""
echo "下一步："
echo "  1. 提交代码到Git: git add . && git commit -m 'update' && git push"
echo "  2. 在远程服务器运行: bash scripts/download_from_huggingface.sh ${HF_USERNAME}"
