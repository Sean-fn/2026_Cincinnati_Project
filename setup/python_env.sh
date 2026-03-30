#!/bin/bash
# Python environment setup script
# Usage: bash setup/python_env.sh [venv|conda]

set -euo pipefail

# ============================================================
# Colored output helpers
# ============================================================
if [ -t 1 ]; then
  BOLD="$(tput bold)"; RESET="$(tput sgr0)"
  GREEN="$(tput setaf 2)"; YELLOW="$(tput setaf 3)"
  BLUE="$(tput setaf 4)"; CYAN="$(tput setaf 6)"
else
  BOLD=""; RESET=""; GREEN=""; YELLOW=""; BLUE=""; CYAN=""
fi

log_step() { echo -e "\n${BOLD}${BLUE}==>${RESET} ${BOLD}$*${RESET}"; }
log_info() { echo -e "${CYAN}[INFO]${RESET} $*"; }
log_ok()   { echo -e "${GREEN}[ OK ]${RESET} $*"; }
log_warn() { echo -e "${YELLOW}[WARN]${RESET} $*"; }
log_err()  { echo -e "${RED}[ERR ]${RESET} $*"; }

ENV_TYPE="${1:-venv}"  # Default: venv

if [ "$ENV_TYPE" = "venv" ]; then
  log_step "Create Python venv"

  if [ ! -d "venv" ]; then
    python3 -m venv venv
  fi

  source venv/bin/activate

  log_step "Upgrade pip/setuptools/wheel"
  pip install --upgrade pip setuptools wheel

  log_step "Install PyTorch (cu121)"
  pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision

  log_step "Install flash-attn deps (psutil)"
  pip install psutil

  log_step "Pin NumPy version (flash-attn requires numpy<2)"
  pip install "numpy<2"

  log_step "Install prebuilt flash-attn wheel"
  # Use prebuilt wheel to avoid requiring CUDA toolkit
  pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.7/flash_attn-2.5.7+cu122torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

  log_step "Install dependencies (requirements.txt, skip flash-attn)"
  pip install -r requirements.txt --no-deps

  log_ok "venv setup complete"
  echo "Activate env: source venv/bin/activate"

elif [ "$ENV_TYPE" = "conda" ]; then
  log_step "Create conda environment"

  if ! command -v conda &>/dev/null; then
    echo "Error: conda not found"
    exit 1
  fi

  conda env create -f environment.yml
  log_ok "Conda environment setup complete"
  echo "Activate env: conda activate M2F2_det"

else
  echo "Error: unknown environment type: $ENV_TYPE"
  echo "Usage: $0 [venv|conda]"
  exit 1
fi
