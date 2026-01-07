#!/bin/bash
# 验证环境和依赖（快速版本，不运行训练）
# 用法: bash scripts/verify_env.sh

set -e

echo "=========================================="
echo "M2F2_Det 环境验证"
echo "=========================================="

# 1. Python版本
echo -e "\n[1/7] Python版本..."
python --version || python3 --version

# 2. 关键依赖
echo -e "\n[2/7] 检查关键依赖..."
python -c "
import sys
from importlib import metadata, util

packages = {
    'torch': ('torch', 'torch'),
    'torchvision': ('torchvision', 'torchvision'),
    'transformers': ('transformers', 'transformers'),
    'deepspeed': ('deepspeed', 'deepspeed'),
    'peft': ('peft', 'peft'),
    'PIL': ('PIL', 'Pillow'),
    'cv2': ('cv2', 'opencv-python'),
}

missing = []
for module_name, dist_name in packages.values():
    if util.find_spec(module_name) is None:
        missing.append(dist_name)
        print(f'✗ {module_name}: NOT INSTALLED')
        continue
    try:
        version = metadata.version(dist_name)
    except metadata.PackageNotFoundError:
        version = 'unknown'
    print(f'✓ {module_name}: {version}')

try:
    numpy_version = metadata.version('numpy')
    major = int(numpy_version.split('.')[0])
    if major >= 2:
        print(f'⚠️  numpy版本为{numpy_version}，部分扩展可能需要降级到numpy<2')
except metadata.PackageNotFoundError:
    print('✗ numpy: NOT INSTALLED')
    missing.append('numpy')

if missing:
    print(f'\n缺少依赖: {missing}')
    print('运行: pip install ' + ' '.join(missing))
    sys.exit(1)
"

# 3. CUDA检查
echo -e "\n[3/7] CUDA支持..."
python -c "
import torch
cuda_available = torch.cuda.is_available()
print(f'CUDA可用: {cuda_available}')
if cuda_available:
    print(f'GPU数量: {torch.cuda.device_count()}')
    print(f'GPU型号: {torch.cuda.get_device_name(0)}')
    print(f'CUDA版本: {torch.version.cuda}')
else:
    print('⚠️  未检测到CUDA（训练需要GPU）')
"

# 4. 检查关键文件
echo -e "\n[4/7] 检查关键文件..."
FILES=(
    "llava/model/deepfake/M2F2Det/model.py"
    "llava/train/train_deepfake.py"
    "llava/model/builder.py"
    "scripts/finetune_kan_qlora.sh"
)

for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "✓ $file"
    else
        echo "✗ 缺失: $file"
        exit 1
    fi
done

# 5. 检查大文件（模型权重）
echo -e "\n[5/7] 检查模型权重..."

if [ -d "checkpoints/llava-1.5-7b-deepfake-rand-proj-v1" ]; then
    echo "✓ checkpoints/llava-1.5-7b-deepfake-rand-proj-v1"
else
    echo "✗ 缺失: checkpoints/llava-1.5-7b-deepfake-rand-proj-v1"
    echo "  运行: bash scripts/download_from_huggingface.sh YOUR_HF_USERNAME"
fi

if [ -f "utils/weights/M2F2_Det_densenet121.pth" ]; then
    echo "✓ utils/weights/M2F2_Det_densenet121.pth"
else
    echo "✗ 缺失: utils/weights/M2F2_Det_densenet121.pth"
    echo "  运行: bash scripts/download_from_huggingface.sh YOUR_HF_USERNAME"
fi

# 6. 检查数据集
echo -e "\n[6/7] 检查训练数据..."

if [ -d "utils/DDVQA_images/c40/train" ]; then
    IMG_COUNT=$(find utils/DDVQA_images/c40/train -type f | wc -l)
    echo "✓ utils/DDVQA_images/c40/train ($IMG_COUNT 张图片)"
else
    echo "✗ 缺失: utils/DDVQA_images/c40/train"
fi

if [ -f "utils/DDVQA_split/c40/train_DDVQA_format.json" ]; then
    echo "✓ utils/DDVQA_split/c40/train_DDVQA_format.json"
else
    echo "✗ 缺失: utils/DDVQA_split/c40/train_DDVQA_format.json"
fi

# 7. 测试导入
echo -e "\n[7/7] 测试Python导入..."
python -c "
import sys
sys.path.insert(0, '.')

try:
    from llava.model.deepfake.M2F2Det.model import M2F2Det
    from llava.model.builder import load_pretrained_model
    from llava.train.train_deepfake import ModelArguments, DataArguments, TrainingArguments
    print('✓ 所有模块导入成功')
except ImportError as e:
    print(f'✗ 导入失败: {e}')
    sys.exit(1)
"

echo ""
echo "=========================================="
echo "✅ 环境验证完成！"
echo "=========================================="
echo ""
echo "下一步："
echo "  - 本地开发: 修改代码后运行此脚本验证"
echo "  - 上传代码: git add . && git commit && git push"
echo "  - 上传权重: bash scripts/upload_to_huggingface.sh YOUR_HF_USERNAME"
echo "  - 远程训练: 克隆仓库 → download_from_huggingface.sh → 开始训练"
