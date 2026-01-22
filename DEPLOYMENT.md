# M2F2_Det 部署指南

本指南说明如何将M2F2_Det从本地开发环境部署到租用的GPU服务器（32GB VRAM）。

## 📋 前置要求

**本地环境**：
- Python 3.10
- Git
- Hugging Face账号（用于存储大文件）

**远程服务器**：
- Ubuntu 22.04
- NVIDIA GPU（32GB+ VRAM）
- CUDA 12.1+
- Docker（推荐）或Python 3.10

---

## 🔄 完整部署流程

### 阶段1：本地开发和验证

#### 1.1 安装依赖

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

#### 1.2 修改代码

正常修改你的代码文件。

#### 1.3 验证环境

```bash
# 快速验证（不运行训练）
bash scripts/utils/verify_env.sh
```

这个脚本会检查：
- ✓ Python依赖是否安装
- ✓ CUDA是否可用
- ✓ 关键代码文件是否存在
- ✓ 模块是否能正常导入

如果验证失败，根据错误提示修复问题。

---

### 阶段2：上传文件

#### 2.1 上传大文件到Hugging Face（仅第一次）

```bash
# 登录Hugging Face
pip install huggingface_hub
huggingface-cli login

# 上传模型权重（~16GB，需要时间）
bash scripts/huggingface/upload_to_huggingface.sh YOUR_HF_USERNAME

# 上传内容：
# - checkpoints/llava-1.5-7b-deepfake-rand-proj-v1/ (14GB)
# - utils/weights/M2F2_Det_densenet121.pth (1.7GB)
```

**注意**：大文件只需上传一次，后续修改代码不需要重新上传。

#### 2.2 提交代码到Git

```bash
# 添加修改的文件
git add .

# 提交
git commit -m "update training config"

# 推送到GitHub
git push origin main
```

**注意**：`.gitignore`已配置排除大文件，只会提交代码和小文件。

---

### 阶段3：远程服务器设置

登录到你的GPU服务器后执行：

#### 3.1 克隆代码仓库

```bash
# 克隆你的仓库
git clone https://github.com/YOUR_USERNAME/M2F2_Det.git
cd M2F2_Det
```

#### 3.2 下载大文件

```bash
# 从Hugging Face下载模型权重
bash scripts/huggingface/download_from_huggingface.sh Sean-fn

# 这会下载：
# - checkpoints/llava-1.5-7b-deepfake-rand-proj-v1/
# - utils/weights/M2F2_Det_densenet121.pth
```

#### 3.3 验证环境

```bash
# 检查所有文件是否就绪
bash scripts/utils/verify_env.sh
```

---

### 阶段4：训练

#### 方案A：使用Docker（推荐）

```bash
# 构建Docker镜像
docker-compose build

# 启动容器并进入
docker-compose run --rm m2f2-dev

# 在容器内运行训练
bash scripts/kan/stage_2_3_combined.sh
```

#### 方案B：直接使用Python环境

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 运行训练
bash scripts/kan/stage_2_3_combined.sh
```

---

## 📁 项目文件结构

```
M2F2_Det/
├── checkpoints/                           # 模型权重（从HF下载）
│   └── llava-1.5-7b-deepfake-rand-proj-v1/
├── utils/
│   ├── weights/
│   │   └── M2F2_Det_densenet121.pth      # Stage-1权重（从HF下载）
│   ├── DDVQA_images/c40/train/           # 训练图片（Git）
│   └── DDVQA_split/c40/*.json            # 训练标注（Git）
├── llava/                                 # 代码（Git）
├── scripts/                               # 脚本（Git）
│   ├── upload_to_huggingface.sh          # 上传大文件
│   ├── download_from_huggingface.sh      # 下载大文件
│   ├── verify_env.sh                     # 验证环境
│   └── finetune_kan_qlora.sh             # 训练脚本
├── Dockerfile                             # Docker配置（Git）
├── docker-compose.yml                     # Docker Compose（Git）
├── requirements.txt                       # Python依赖（Git）
└── DEPLOYMENT.md                          # 本文档（Git）
```

---

## 🔧 32GB VRAM优化配置

`scripts/kan/stage_2_3_combined.sh` 已针对32GB VRAM优化：

```bash
--bits 4                           # 4-bit量化（节省75%显存）
--per_device_train_batch_size 8    # Batch size
--gradient_accumulation_steps 20   # 梯度累积
--gradient_checkpointing True      # 梯度检查点（节省50%）
--bf16 True                        # BF16混合精度
```

**预估显存使用**：
- 模型加载（4-bit）：~3.5GB
- KAN adapter：~0.5GB
- LoRA参数：~2GB
- Activations：~20GB
- 优化器：~5GB
- **总计：~31GB** ✓

如果仍然OOM，可以降低batch size：
```bash
--per_device_train_batch_size 4    # 降低到4
--gradient_accumulation_steps 40   # 相应增加累积步数
```

---

## 🚀 常见工作流

### 场景1：修改代码后重新训练

```bash
# 本地
vim llava/model/xxx.py
bash scripts/utils/verify_env.sh
git add . && git commit -m "fix bug" && git push

# 远程
cd M2F2_Det
git pull
bash scripts/kan/stage_2_3_combined.sh
```

### 场景2：更新模型权重

```bash
# 本地
# （训练完成后得到新权重）
bash scripts/huggingface/upload_to_huggingface.sh YOUR_HF_USERNAME

# 远程
cd M2F2_Det
bash scripts/huggingface/download_from_huggingface.sh YOUR_HF_USERNAME
```

### 场景3：从零开始部署

```bash
# 本地（首次）
git clone https://github.com/YOUR_USERNAME/M2F2_Det.git
cd M2F2_Det
bash scripts/utils/verify_env.sh
bash scripts/huggingface/upload_to_huggingface.sh YOUR_HF_USERNAME
git add . && git commit -m "init" && git push

# 远程（首次）
git clone https://github.com/YOUR_USERNAME/M2F2_Det.git
cd M2F2_Det
bash scripts/huggingface/download_from_huggingface.sh YOUR_HF_USERNAME
bash scripts/utils/verify_env.sh
bash scripts/kan/stage_2_3_combined.sh
```

---

## ❓ 故障排除

### 问题1: Hugging Face上传/下载失败

**解决方法**：
```bash
# 检查登录状态
huggingface-cli whoami

# 重新登录
huggingface-cli login

# 手动下载（如果脚本失败）
python -c "
from huggingface_hub import snapshot_download
snapshot_download('YOUR_USERNAME/M2F2-Det-Weights', local_dir='./checkpoints')
"
```

### 问题2: CUDA Out of Memory

**解决方法**：
```bash
# 降低batch size（在 finetune_kan_qlora.sh 中）
--per_device_train_batch_size 4
--gradient_accumulation_steps 40
```

### 问题3: 模块导入失败

**解决方法**：
```bash
# 检查PYTHONPATH
export PYTHONPATH="$(pwd):$PYTHONPATH"

# 重新安装依赖
pip install -r requirements.txt
```

### 问题4: Docker build失败

**解决方法**：
```bash
# 使用本地Python环境代替Docker
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 📞 获取帮助

- **项目文档**: [README.md](README.md)
- **技术细节**: [CLAUDE.md](CLAUDE.md)
- **原始论文**: CVPR 2025 - Multi-Modal Interpretable Forged Face Detector

---

## ✅ 检查清单

部署前确认：

- [ ] 本地环境验证通过（`verify_env.sh`）
- [ ] 大文件已上传到Hugging Face
- [ ] 代码已推送到Git
- [ ] 远程服务器已克隆仓库
- [ ] 远程已下载大文件
- [ ] 远程环境验证通过

开始训练：

- [ ] GPU可用（`nvidia-smi`）
- [ ] 所有数据文件就绪
- [ ] 训练脚本参数正确
- [ ] 输出目录可写

---

**祝训练顺利！** 🎉
