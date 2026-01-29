#!/bin/bash
# Environment setup for M2F2-Det
# Source this file before running training scripts

# Get project root directory
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}==> Setting up M2F2-Det environment...${NC}"

# ============================================================
# Detect Python environment (venv or conda)
# ============================================================
if [ -d "$PROJECT_ROOT/venv" ]; then
    # venv environment
    ENV_TYPE="venv"

    # Activate venv if not already active
    if [ -z "$VIRTUAL_ENV" ]; then
        source "$PROJECT_ROOT/venv/bin/activate"
        echo -e "${GREEN}✓${NC} Activated venv: $PROJECT_ROOT/venv"
    else
        echo -e "${GREEN}✓${NC} venv already active: $VIRTUAL_ENV"
    fi

    # Set LD_LIBRARY_PATH for PyTorch cu121 in venv
    PYTHON_VER=$(python -c "import sys; print(f'python{sys.version_info.major}.{sys.version_info.minor}')")
    VENV_SITE_PACKAGES="$PROJECT_ROOT/venv/lib/$PYTHON_VER/site-packages"

    # PyTorch and NVIDIA CUDA libraries (包含所有必要的库)
    export LD_LIBRARY_PATH=\
${VENV_SITE_PACKAGES}/torch/lib:\
${VENV_SITE_PACKAGES}/nvidia/cublas/lib:\
${VENV_SITE_PACKAGES}/nvidia/cudnn/lib:\
${VENV_SITE_PACKAGES}/nvidia/cuda_runtime/lib:\
${VENV_SITE_PACKAGES}/nvidia/cuda_cupti/lib:\
${VENV_SITE_PACKAGES}/nvidia/cuda_nvrtc/lib:\
${VENV_SITE_PACKAGES}/nvidia/cusparse/lib:\
${VENV_SITE_PACKAGES}/nvidia/cusolver/lib:\
${VENV_SITE_PACKAGES}/nvidia/curand/lib:\
${VENV_SITE_PACKAGES}/nvidia/cufft/lib:\
${VENV_SITE_PACKAGES}/nvidia/nccl/lib:\
${VENV_SITE_PACKAGES}/nvidia/nvjitlink/lib:\
${VENV_SITE_PACKAGES}/nvidia/nvtx/lib:\
${LD_LIBRARY_PATH}

    echo -e "${GREEN}✓${NC} PyTorch CUDA 12.1 libraries configured"

else
    ENV_TYPE="unknown"
    echo -e "${YELLOW}⚠${NC} venv not found at $PROJECT_ROOT/venv"
    echo "  Please create the environment first:"
    echo "    bash setup/python_env.sh venv"
    return 1
fi

# ============================================================
# Set PYTHONPATH
# ============================================================
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
echo -e "${GREEN}✓${NC} PYTHONPATH configured"

# ============================================================
# Verify PyTorch installation
# ============================================================
echo ""
echo -e "${CYAN}Environment verification:${NC}"

if python -c "import torch" 2>/dev/null; then
    PYTORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
    CUDA_AVAILABLE=$(python -c "import torch; print(torch.cuda.is_available())")
    CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda if torch.cuda.is_available() else 'N/A')")

    echo -e "  ${GREEN}✓${NC} PyTorch: $PYTORCH_VERSION"
    echo -e "  ${GREEN}✓${NC} CUDA available: $CUDA_AVAILABLE"
    echo -e "  ${GREEN}✓${NC} CUDA version: $CUDA_VERSION"
else
    echo -e "  ${YELLOW}⚠${NC} PyTorch not found or failed to import"
fi

echo ""
echo -e "${GREEN}==> Environment setup complete!${NC}"
echo ""
