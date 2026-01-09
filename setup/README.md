# M2F2_Det 环境设置工具

本目录包含用于设置 M2F2_Det 环境和下载预训练权重的模块化脚本。

## 文件说明

### 1. `download_config.sh` - 下载配置文件
集中管理所有下载相关的配置参数。

**关键配置**:
- `DOWNLOAD_STAGE1_WEIGHTS`: 是否下载 Stage-1 检测器权重 (1.7GB)
- `DOWNLOAD_STAGE2_INIT_WEIGHTS`: 是否下载 Stage-2 初始化权重 (14GB)
- `DOWNLOAD_LLAVA_BASE`: 是否下载 LLaVA-1.5-7b 基础模型 (13GB)
- `DOWNLOAD_INFERENCE_MODEL`: 是否下载推理模型 (14GB)
- `DOWNLOAD_CLIP_ENCODER`: 是否下载 CLIP 视觉编码器 (400MB)
- `DOWNLOAD_DDVQA_DATASET`: 是否下载/解压 DDVQA 数据集

### 2. `download_weights.sh` - 权重下载脚本
自动化下载所有配置的模型权重和数据集。

**使用方式**:
```bash
# 使用默认配置下载
bash setup/download_weights.sh

# 使用自定义配置
bash setup/download_weights.sh --config my_config.sh

# 静默模式 (用于被其他脚本调用)
bash setup/download_weights.sh --quiet
```

**功能**:
- ✓ 从 Hugging Face 下载权重
- ✓ 支持断点续传
- ✓ 自动创建目录结构
- ✓ 验证下载完整性
- ✓ 彩色输出和进度提示

### 3. `python_env.sh` - Python环境配置
创建并配置 Python 虚拟环境。

**使用方式**:
```bash
# 使用 venv (默认)
bash setup/python_env.sh venv

# 使用 conda
bash setup/python_env.sh conda
```

### 4. `install_cuda.sh` - CUDA 11.7 安装脚本
自动安装 CUDA 11.7 toolkit,配置环境变量,并可选安装 Flash Attention。

**使用方式**:
```bash
# 自动安装 CUDA 11.7
bash setup/install_cuda.sh
```

**功能**:
- ✓ 检测并安装 CUDA 11.7
- ✓ 自动配置环境变量 (CUDA_HOME, PATH, LD_LIBRARY_PATH)
- ✓ 添加配置到 ~/.bashrc
- ✓ 可选安装 Flash Attention 2.5.7
- ✓ 支持 Ubuntu 20.04 和 22.04

**重要提示**:
- CUDA 11.7 是训练所必需的 (Flash Attention 要求 CUDA 11.6+)
- 安装后需要重新登录或 `source ~/.bashrc`
- Flash Attention 编译需要 5-10 分钟

## 快速开始

### 场景1: 训练配置
仅下载训练所需的权重:

```bash
# 1. 编辑配置文件
cat > setup/download_config_train.sh <<EOF
DOWNLOAD_STAGE1_WEIGHTS=true
DOWNLOAD_STAGE2_INIT_WEIGHTS=true
DOWNLOAD_LLAVA_BASE=false
DOWNLOAD_INFERENCE_MODEL=false
DOWNLOAD_CLIP_ENCODER=false
DOWNLOAD_DDVQA_DATASET=true
source setup/download_config.sh
EOF

# 2. 下载权重
bash setup/download_weights.sh --config setup/download_config_train.sh
```

### 场景2: 推理配置
仅下载推理所需的模型:

```bash
# 修改 setup/download_config.sh
# 设置 DOWNLOAD_INFERENCE_MODEL=true
# 其他设为 false

bash setup/download_weights.sh
```

### 场景3: 完整设置
下载所有权重和数据集:

```bash
# 1. 安装 CUDA 11.7
bash setup/install_cuda.sh

# 2. 设置 Python 环境
bash setup/python_env.sh venv
source venv/bin/activate

# 3. 下载所有权重 (编辑配置文件,全部设为 true)
bash setup/download_weights.sh

# 4. 验证环境
bash scripts/verify_env.sh
```

### 场景4: DDVQA 数据集
DDVQA 数据集处理:

```bash
# 方案1: 如果已有 c40.zip
# 确保 c40.zip 在 utils/DDVQA_images/ 目录下
# 运行下载脚本会自动解压
bash setup/download_weights.sh

# 方案2: 从 GitHub 下载
# 访问: https://github.com/Reality-Defender/Research-DD-VQA
# 下载 c40.zip 后放到 utils/DDVQA_images/
```

## 配置示例

### 训练配置
```bash
DOWNLOAD_STAGE1_WEIGHTS=true
DOWNLOAD_STAGE2_INIT_WEIGHTS=true
DOWNLOAD_LLAVA_BASE=false      # Stage-2 权重已包含
DOWNLOAD_INFERENCE_MODEL=false
DOWNLOAD_CLIP_ENCODER=false
DOWNLOAD_DDVQA_DATASET=true
```

### 推理配置
```bash
DOWNLOAD_STAGE1_WEIGHTS=false
DOWNLOAD_STAGE2_INIT_WEIGHTS=false
DOWNLOAD_LLAVA_BASE=false
DOWNLOAD_INFERENCE_MODEL=true   # 仅需推理模型
DOWNLOAD_CLIP_ENCODER=false
DOWNLOAD_DDVQA_DATASET=true
```

### 开发配置
```bash
DOWNLOAD_STAGE1_WEIGHTS=true
DOWNLOAD_STAGE2_INIT_WEIGHTS=true
DOWNLOAD_LLAVA_BASE=true
DOWNLOAD_INFERENCE_MODEL=true
DOWNLOAD_CLIP_ENCODER=true
DOWNLOAD_DDVQA_DATASET=true
```

## 下载说明

### Hugging Face 仓库
- **训练权重**: `Sean-fn/M2F2-Det-Weights`
  - Stage-1 检测器 (1.7GB)
  - Stage-2 初始化权重 (14GB)

- **推理模型**: `CHELSEA234/llava-v1.5-7b-M2F2-Det` (14GB)

- **LLaVA 基础**: `liuhaotian/llava-v1.5-7b` (13GB)

### 存储需求
- **最小配置** (仅训练): ~16GB
- **推理配置**: ~14GB
- **完整配置**: ~43GB

## 故障排除

### 下载失败
```bash
# 检查网络连接
ping huggingface.co

# 手动安装 huggingface_hub
pip install huggingface_hub

# 重新运行下载 (支持断点续传)
bash setup/download_weights.sh
```

### 权限错误
```bash
# 添加执行权限
chmod +x setup/*.sh
```

### Python 环境问题
```bash
# 检查 Python 版本
python3 --version  # 需要 Python 3.10+

# 手动创建 venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### CUDA 相关问题

#### CUDA 11.7 安装失败
```bash
# 检查系统版本
lsb_release -a

# 手动安装步骤
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-11-7
```

#### Flash Attention 编译失败
```bash
# 确保 CUDA 环境变量正确
export CUDA_HOME=/usr/local/cuda-11.7
export PATH=/usr/local/cuda-11.7/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH

# 安装 ninja (加速编译)
sudo apt-get install -y ninja-build

# 重新安装 flash-attn
pip install flash-attn==2.5.7 --no-build-isolation
```

#### 验证 CUDA 安装
```bash
# 检查 nvcc
nvcc --version

# 检查 CUDA 目录
ls -la /usr/local/cuda-11.7

# 检查环境变量
echo $CUDA_HOME
echo $PATH | tr ':' '\n' | grep cuda
```

### DDVQA 数据集问题

#### c40.zip 不存在
```bash
# 选项1: 从 GitHub 下载
git clone https://github.com/Reality-Defender/Research-DD-VQA.git
# 从克隆的仓库中找到 c40.zip

# 选项2: 手动放置
# 将 c40.zip 放到 utils/DDVQA_images/ 目录
# 然后运行: bash setup/download_weights.sh
```

#### 解压失败
```bash
# 手动解压
unzip utils/DDVQA_images/c40.zip -d utils/DDVQA_images/

# 验证解压
ls -la utils/DDVQA_images/c40/
```

## 与 init.sh 集成

这些脚本已集成到 `init.sh` 中:
- `init.sh` 会自动调用 `setup/python_env.sh` 创建环境
- `init.sh` 会自动调用 `setup/download_weights.sh --quiet` 下载权重
- 如果脚本不存在,会使用 fallback 方法

## 更多信息

- 查看主 README: `../README.md`
- 查看项目文档: `../CLAUDE.md`
- 环境验证: `bash scripts/verify_env.sh`
