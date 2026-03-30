# M2F2_Det Environment Setup Tools

This directory contains modular scripts for setting up the M2F2_Det environment and downloading pretrained weights.

## Files

### 1. `download_config.sh` - Download configuration
Centralized configuration for all download options.

**Key options**:
- `DOWNLOAD_STAGE1_WEIGHTS`: Download Stage-1 detector weights (1.7GB)
- `DOWNLOAD_STAGE2_INIT_WEIGHTS`: Download Stage-2 init weights (14GB)
- `DOWNLOAD_LLAVA_BASE`: Download LLaVA-1.5-7b base model (13GB)
- `DOWNLOAD_INFERENCE_MODEL`: Download inference model (14GB)
- `DOWNLOAD_CLIP_ENCODER`: Download CLIP vision encoder (400MB)
- `DOWNLOAD_DDVQA_DATASET`: Download/unpack DDVQA dataset

### 2. `download_weights.sh` - Weights download script
Automates downloading all configured model weights and datasets.

**Usage**:

```bash
# Download with default config
bash setup/download_weights.sh

# Download with custom config
bash setup/download_weights.sh --config /path/to/custom_config.sh

# Quiet mode (for other scripts)
bash setup/download_weights.sh --quiet
```

**Features**:
- ✓ Download weights from Hugging Face
- ✓ Resume interrupted downloads
- ✓ Auto-create directory structure
- ✓ Verify download completeness
- ✓ Colored output and progress

### 3. `python_env.sh` - Python environment setup
Creates and configures a Python virtual environment.

**Usage**:

```bash
# Use venv (default)
bash setup/python_env.sh

# Use conda
bash setup/python_env.sh conda
```

### 4. `install_cuda.sh` - CUDA 12.1 installer
Automatically installs CUDA 12.1 toolkit, configures environment variables, and optionally installs Flash Attention.

**Usage**:

```bash
# Install CUDA 12.1
bash setup/install_cuda.sh
```

**Features**:
- ✓ Detects and installs CUDA 12.1
- ✓ Configures environment variables (CUDA_HOME, PATH, LD_LIBRARY_PATH)
- ✓ Adds config to ~/.bashrc
- ✓ Optional Flash Attention 2.5.7 install
- ✓ Supports Ubuntu 20.04 and 22.04

**Important**:
- CUDA 12.1 is required for training (Flash Attention requires CUDA 12.1+)
- Re-login or `source ~/.bashrc` after installation
- Flash Attention build may take 5-10 minutes

## Quickstart

### Scenario 1: Training setup
Download only training weights:

```bash
# 1. Edit config file
nano setup/download_config.sh

# 2. Download weights
bash setup/download_weights.sh
```

### Scenario 2: Inference setup
Download only inference model:

```bash
# Edit setup/download_config.sh
# Set DOWNLOAD_INFERENCE_MODEL=true
# Set others to false
bash setup/download_weights.sh
```

### Scenario 3: Full setup
Download all weights and datasets:

```bash
# 1. Install CUDA 12.1
bash setup/install_cuda.sh

# 2. Set up Python environment
bash setup/python_env.sh

# 3. Download all weights (set all to true in config)
bash setup/download_weights.sh

# 4. Verify environment
bash scripts/verify_env.sh
```

### Scenario 4: DDVQA dataset
DDVQA dataset handling:

```bash
# Option 1: If you already have c40.zip
# Ensure c40.zip is under utils/DDVQA_images/
# Running download script will unzip it automatically

# Option 2: Download from GitHub
# Visit: https://github.com/Reality-Defender/Research-DD-VQA
# Download c40.zip and place it in utils/DDVQA_images/
```

## Configuration examples

### Training config
```bash
DOWNLOAD_STAGE1_WEIGHTS=true
DOWNLOAD_STAGE2_INIT_WEIGHTS=true
DOWNLOAD_LLAVA_BASE=false      # Stage-2 weights already include it
DOWNLOAD_INFERENCE_MODEL=false
DOWNLOAD_CLIP_ENCODER=false
DOWNLOAD_DDVQA_DATASET=true
```

### Inference config
```bash
DOWNLOAD_STAGE1_WEIGHTS=false
DOWNLOAD_STAGE2_INIT_WEIGHTS=false
DOWNLOAD_LLAVA_BASE=false
DOWNLOAD_INFERENCE_MODEL=true   # Only need inference model
DOWNLOAD_CLIP_ENCODER=false
DOWNLOAD_DDVQA_DATASET=false
```

### Development config
```bash
DOWNLOAD_STAGE1_WEIGHTS=true
DOWNLOAD_STAGE2_INIT_WEIGHTS=true
DOWNLOAD_LLAVA_BASE=true
DOWNLOAD_INFERENCE_MODEL=true
DOWNLOAD_CLIP_ENCODER=true
DOWNLOAD_DDVQA_DATASET=true
```

## Download details

### Hugging Face repositories
- **Training weights**: `Sean-fn/M2F2-Det-Weights`
  - Stage-1 detector (1.7GB)
  - Stage-2 init weights (14GB)

- **Inference model**: `CHELSEA234/llava-v1.5-7b-M2F2-Det` (14GB)

- **LLaVA base**: `liuhaotian/llava-v1.5-7b` (13GB)

### Storage requirements
- **Minimal (training only)**: ~16GB
- **Inference only**: ~14GB
- **Full setup**: ~43GB

## Troubleshooting

### Download failures
```bash
# Check network connectivity

# Manually install huggingface_hub
pip install huggingface_hub

# Re-run download (resumable)
bash setup/download_weights.sh
```

### Permission errors
```bash
# Add execute permissions
chmod +x setup/*.sh
```

### Python environment issues
```bash
# Check Python version
python3 --version  # Requires Python 3.10+

# Manually create venv
python3 -m venv venv
```

### CUDA issues

#### CUDA 12.1 install failure
```bash
# Check system version
cat /etc/os-release

# Manual installation steps
# See NVIDIA CUDA installation docs
```

#### Flash Attention build failure
```bash
# Ensure CUDA environment variables are correct
export CUDA_HOME=/usr/local/cuda-12.1
export PATH=/usr/local/cuda-12.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH

# Install ninja (faster build)
sudo apt-get install -y ninja-build

# Reinstall flash-attn
pip install flash-attn==2.5.7 --no-build-isolation
```

#### Verify CUDA installation
```bash
# Check nvcc
nvcc --version

# Check CUDA directory
ls /usr/local/cuda-12.1

# Check environment variables
echo $CUDA_HOME
echo $PATH
echo $LD_LIBRARY_PATH
```

### DDVQA dataset issues

#### c40.zip not found
```bash
# Option 1: Download from GitHub
# Find c40.zip in the cloned repo

# Option 2: Place manually
# Put c40.zip into utils/DDVQA_images/
# Then run: bash setup/download_weights.sh
```

#### Unzip failure
```bash
# Manual unzip
unzip c40.zip -d utils/DDVQA_images/

# Verify
ls utils/DDVQA_images/c40
```

## Integration with init.sh
These scripts are integrated into `init.sh`:
- `init.sh` will call `setup/python_env.sh` to create the environment
- `init.sh` will call `setup/download_weights.sh --quiet` to download weights
- If scripts are missing, it uses fallback methods

## More info
- Main README: `../README.md`
- Project docs: `../CLAUDE.md`
- Environment verification: `bash scripts/verify_env.sh`
