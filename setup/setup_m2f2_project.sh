#!/bin/bash
# ============================================================
# M2F2_Det project environment setup script
#
# Features:
#   1. Create Python virtual environment (venv or conda)
#   2. Install Python dependencies (requirements.txt)
#   3. Download pretrained weights (Stage-1, Stage-2, LLaVA, etc.)
#   4. Download DDVQA dataset
#   5. Install CUDA 12.1 (optional but recommended)
#   6. Verify environment configuration
#
# Usage:
#   bash setup/setup_m2f2_project.sh [--env venv|conda] [--skip-cuda] [--skip-download]
# ============================================================

set -euo pipefail

# ============================================================
# Colored output helpers
# ============================================================
if [ -t 1 ]; then
  BOLD="$(tput bold)"; DIM="$(tput dim)"; RESET="$(tput sgr0)"
  RED="$(tput setaf 1)"; GREEN="$(tput setaf 2)"; YELLOW="$(tput setaf 3)"
  BLUE="$(tput setaf 4)"; CYAN="$(tput setaf 6)"
else
  BOLD=""; DIM=""; RESET=""
  RED=""; GREEN=""; YELLOW=""; BLUE=""; CYAN=""
fi

_ts() { date +"%H:%M:%S"; }

log_step() { echo -e "\n${BOLD}${BLUE}==>${RESET} ${BOLD}$*${RESET}"; }
log_info() { echo -e "${CYAN}[INFO $(_ts)]${RESET} $*"; }
log_ok()   { echo -e "${GREEN}[ OK  $(_ts)]${RESET} $*"; }
log_warn() { echo -e "${YELLOW}[WARN $(_ts)]${RESET} $*"; }
log_err()  { echo -e "${RED}[ERR  $(_ts)]${RESET} $*"; }

run() {
  local label="$1"; shift
  if "$@" >/dev/null 2>&1; then
    log_ok "$label"
  else
    local code=$?
    log_err "$label (exit=${code})"
    return "$code"
  fi
}

run_spin() {
  local label="$1"; shift
  local spinner='|/-\'
  local i=0
  ("$@" >/dev/null 2>&1) &
  local pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    printf "\r${DIM}%s %s...${RESET}" "${spinner:i++%4:1}" "$label"
    sleep 0.1
  done
  if wait "$pid"; then
    printf "\r"
    log_ok "$label"
  else
    local c=$?
    printf "\r"
    log_err "$label (exit=$c)"
    return "$c"
  fi
}

trap 'log_err "Failed at line $LINENO: $BASH_COMMAND"' ERR

# ============================================================
# Argument parsing
# ============================================================
ENV_TYPE="venv"          # Default: venv
SKIP_CUDA=false
SKIP_DOWNLOAD=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --env)
      ENV_TYPE="$2"
      shift 2
      ;;
    --skip-cuda)
      SKIP_CUDA=true
      shift
      ;;
    --skip-download)
      SKIP_DOWNLOAD=true
      shift
      ;;
    -h|--help)
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --env venv|conda    Choose environment type (default: venv)"
      echo "  --skip-cuda         Skip CUDA 12.1 installation"
      echo "  --skip-download     Skip weights and dataset download"
      echo "  -h, --help          Show help"
      echo ""
      echo "Examples:"
      echo "  $0                          # Use default configuration"
      echo "  $0 --env conda              # Use conda environment"
      echo "  $0 --skip-cuda              # Skip CUDA installation"
      echo "  $0 --env venv --skip-download  # Setup env only, no downloads"
      exit 0
      ;;
    *)
      log_err "Unknown option: $1"
      echo "Use --help to see options"
      exit 1
      ;;
  esac
done

# ============================================================
# Locate project root
# ============================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

log_info "Project root: $PROJECT_ROOT"
cd "$PROJECT_ROOT"

# ============================================================
# Banner
# ============================================================
echo ""
echo "========================================"
echo "M2F2_Det Project Environment Setup"
echo "========================================"
echo ""
log_info "Configuration:"
echo "  - Python env: $ENV_TYPE"
echo "  - Skip CUDA: $SKIP_CUDA"
echo "  - Skip download: $SKIP_DOWNLOAD"
echo ""

# ============================================================
# 1. Create Python environment
# ============================================================
log_step "1/6 Create Python environment ($ENV_TYPE)"

if [ -f "setup/python_env.sh" ]; then
  run_spin "setup Python environment" bash setup/python_env.sh "$ENV_TYPE"
else
  log_warn "setup/python_env.sh not found, using fallback"

  if [ "$ENV_TYPE" = "venv" ]; then
    if [ -d "venv" ]; then
      log_ok "venv already exists"
    else
      run_spin "create venv" python3 -m venv venv
    fi

    log_info "Activating venv..."
    source venv/bin/activate

    run_spin "upgrade pip" pip install --upgrade pip
    run_spin "install requirements" pip install -r requirements.txt
  elif [ "$ENV_TYPE" = "conda" ]; then
    if ! command -v conda &>/dev/null; then
      log_err "conda not found, install Anaconda/Miniconda first"
      exit 1
    fi

    run_spin "create conda env" conda env create -f environment.yml
    log_ok "Conda environment created"
    log_info "Activate manually: conda activate M2F2_det"
  else
    log_err "Unsupported environment type: $ENV_TYPE"
    exit 1
  fi
fi

# ============================================================
# 2. Download pretrained weights and datasets
# ============================================================
if [ "$SKIP_DOWNLOAD" = false ]; then
  log_step "2/6 Download pretrained weights and datasets"

  if [ -f "setup/download_weights.sh" ]; then
    bash setup/download_weights.sh
  else
    log_warn "setup/download_weights.sh not found"

    if [ -f "scripts/download_from_huggingface.sh" ]; then
      log_info "Using legacy download script"
      run_spin "download weights" bash scripts/download_from_huggingface.sh "Sean-fn"
    else
      log_err "No download script found"
    fi
  fi
else
  log_step "2/6 Skip downloads (--skip-download)"
fi

# ============================================================
# 3. Install CUDA 12.1
# ============================================================
if [ "$SKIP_CUDA" = false ]; then
  log_step "3/6 Install CUDA 12.1"

  if [ -d "/usr/local/cuda-12.1" ]; then
    log_ok "CUDA 12.1 is installed"

    # Check environment variables
    if [ -n "${CUDA_HOME:-}" ] && [ "$CUDA_HOME" = "/usr/local/cuda-12.1" ]; then
      log_ok "CUDA_HOME is configured"
    else
      log_warn "CUDA_HOME not configured. Run: export CUDA_HOME=/usr/local/cuda-12.1"
    fi
  else
    log_warn "CUDA 12.1 not installed"

    if [ -f "setup/install_cuda.sh" ]; then
      log_info "CUDA 12.1 is required for training (Flash Attention requirement)"
      bash setup/install_cuda.sh
    else
      log_warn "setup/install_cuda.sh not found"
    fi
  fi
else
  log_step "3/6 Skip CUDA 12.1 installation (--skip-cuda)"
fi

# ============================================================
# 4. Verify environment
# ============================================================
log_step "4/6 Verify environment"

if [ -f "scripts/verify_env.sh" ]; then
  bash scripts/verify_env.sh
else
  log_warn "scripts/verify_env.sh not found, skipping verification"

  # Minimal checks
  log_info "Basic checks:"

  if command -v python3 &>/dev/null; then
    log_ok "Python: $(python3 --version)"
  else
    log_err "Python3 not found"
  fi

  if python3 -c "import torch" 2>/dev/null; then
    log_ok "PyTorch is installed"
  else
    log_warn "PyTorch not installed or failed to import"
  fi

  if python3 -c "import transformers" 2>/dev/null; then
    log_ok "Transformers is installed"
  else
    log_warn "Transformers not installed or failed to import"
  fi
fi

# ============================================================
# 5. Check key files
# ============================================================
log_step "5/6 Check key files and directories"

check_path() {
  local path="$1"
  local name="$2"

  if [ -e "$path" ]; then
    log_ok "$name: $path"
  else
    log_warn "Missing $name: $path"
  fi
}

check_path "checkpoints/llava-1.5-7b-deepfake-rand-proj-v1" "Stage-2 init weights"
check_path "utils/weights/M2F2_Det_densenet121.pth" "Stage-1 detector"
check_path "utils/DDVQA_images/c40" "DDVQA dataset"
check_path "requirements.txt" "Python requirements"
check_path "environment.yml" "Conda environment config"

# ============================================================
# 6. Summary
# ============================================================
log_step "6/6 Setup complete"

echo ""
echo "========================================"
echo -e "${BOLD}${GREEN}✅ M2F2_Det environment setup complete!${RESET}"
echo "========================================"
echo ""

log_info "Environment:"
if [ "$ENV_TYPE" = "venv" ]; then
  echo "  - Python env: venv"
  echo "  - Activate: ${BOLD}source venv/bin/activate${RESET}"
elif [ "$ENV_TYPE" = "conda" ]; then
  echo "  - Python env: conda"
  echo "  - Activate: ${BOLD}conda activate M2F2_det${RESET}"
fi

echo ""
log_info "Training:"
echo "  1) Stage-1 training: ${BOLD}bash stage_1_train.sh${RESET}"
echo "  2) Stage-2 training: ${BOLD}bash stage_2_train.sh${RESET}"
echo "  3) Stage-3 training: ${BOLD}bash stage_3_train.sh${RESET}"

echo ""
log_info "Inference:"
echo "  1) Detection inference: ${BOLD}bash stage_3_inference_det.sh${RESET}"
echo "  2) Explanation generation: ${BOLD}bash stage_3_inference_exp.sh${RESET}"
echo "  3) Evaluate results: ${BOLD}python eval/eval_judgement.py${RESET}"

echo ""
log_info "Low VRAM training (KAN + QLoRA):"
echo "  - Run: ${BOLD}bash scripts/kan/stage_2_3_combined.sh${RESET}"

echo ""
log_info "Docs:"
echo "  - Quickstart: ${BOLD}QUICKSTART.md${RESET}"
echo "  - Setup docs: ${BOLD}setup/README.md${RESET}"
echo "  - Project docs: ${BOLD}CLAUDE.md${RESET}"

if [ "$SKIP_CUDA" = false ] && [ ! -d "/usr/local/cuda-12.1" ]; then
  echo ""
  log_warn "Reminder: CUDA 12.1 is not installed"
  echo "  - Flash Attention installed (prebuilt wheel, cu122)"
  echo "  - Training requires CUDA 12.1+ runtime for Flash Attention"
  echo "  - Install: ${BOLD}bash setup/install_cuda.sh${RESET}"
fi

echo ""
