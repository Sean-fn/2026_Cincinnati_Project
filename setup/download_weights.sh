#!/bin/bash
# M2F2_Det weights download script
# Usage: bash setup/download_weights.sh [--config FILE] [--quiet]

set -euo pipefail

# ============================================================
# Argument parsing
# ============================================================
CONFIG_FILE="setup/download_config.sh"
QUIET_MODE=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --config) CONFIG_FILE="$2"; shift 2 ;;
    --quiet)  QUIET_MODE=true; shift ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

# ============================================================
# Load configuration
# ============================================================
if [ ! -f "$CONFIG_FILE" ]; then
  echo "Error: config file not found: $CONFIG_FILE"
  exit 1
fi

source "$CONFIG_FILE"

# ============================================================
# Colored output helpers
# ============================================================
if [ "$QUIET_MODE" = false ] && [ -t 1 ]; then
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

# ============================================================
# Download helpers
# ============================================================

# Check dependencies
check_dependencies() {
  log_step "Checking download dependencies"

  if ! command -v python3 &>/dev/null; then
    echo "Error: python3 not found"
    exit 1
  fi

  # Install huggingface_hub
  if ! python3 -c "import huggingface_hub" &>/dev/null 2>&1; then
    log_info "Installing huggingface_hub..."
    pip install -q huggingface_hub
  fi

  log_ok "Dependency check complete"
}

# Create directory structure
create_directories() {
  log_step "Creating directory structure"
  mkdir -p "$CHECKPOINT_DIR"
  mkdir -p "$WEIGHTS_DIR"
  mkdir -p "$DATASET_DIR"
  log_ok "Directories created"
}

# Download Stage-1 detector weights
download_stage1_weights() {
  if [ "$DOWNLOAD_STAGE1_WEIGHTS" = false ]; then
    log_info "Skipping Stage-1 weights"
    return
  fi

  log_step "Downloading Stage-1 detector weights (1.7GB)"

  python3 - <<EOF_PY
from huggingface_hub import hf_hub_download
import os

try:
    hf_hub_download(
        repo_id="${HF_TRAINING_REPO}",
        filename="weights/M2F2_Det_densenet121.pth",
        local_dir="${WEIGHTS_DIR%/*}",
        local_dir_use_symlinks=False
    )
    print("✓ Stage-1 weights downloaded")
except Exception as e:
    print(f"❌ Download failed: {e}")
    exit(1)
EOF_PY

  log_ok "Stage-1 weights: ${WEIGHTS_DIR}/M2F2_Det_densenet121.pth"
}

# Download Stage-2 init weights
download_stage2_init_weights() {
  if [ "$DOWNLOAD_STAGE2_INIT_WEIGHTS" = false ]; then
    log_info "Skipping Stage-2 init weights"
    return
  fi

  log_step "Downloading Stage-2 init weights (14GB, may take a while)"

  python3 - <<EOF_PY
from huggingface_hub import snapshot_download
import os

try:
    snapshot_download(
        repo_id="${HF_TRAINING_REPO}",
        allow_patterns="llava-1.5-7b-deepfake-rand-proj-v1/*",
        local_dir="${CHECKPOINT_DIR}",
        local_dir_use_symlinks=False,
        resume_download=True
    )
    print("✓ Stage-2 init weights downloaded")
except Exception as e:
    print(f"❌ Download failed: {e}")
    exit(1)
EOF_PY

  log_ok "Stage-2 weights: ${CHECKPOINT_DIR}/llava-1.5-7b-deepfake-rand-proj-v1/"
}

# Download LLaVA base model
download_llava_base() {
  if [ "$DOWNLOAD_LLAVA_BASE" = false ]; then
    log_info "Skipping LLaVA base model"
    return
  fi

  log_step "Downloading LLaVA-1.5-7b base model (13GB)"

  python3 - <<EOF_PY
from huggingface_hub import snapshot_download

try:
    snapshot_download(
        repo_id="${HF_LLAVA_BASE_REPO}",
        local_dir="${CHECKPOINT_DIR}/llava-v1.5-7b",
        local_dir_use_symlinks=False,
        resume_download=True
    )
    print("✓ LLaVA base model downloaded")
except Exception as e:
    print(f"❌ Download failed: {e}")
    exit(1)
EOF_PY

  log_ok "LLaVA base model: ${CHECKPOINT_DIR}/llava-v1.5-7b/"
}

# Download inference model
download_inference_model() {
  if [ "$DOWNLOAD_INFERENCE_MODEL" = false ]; then
    log_info "Skipping inference model"
    return
  fi

  log_step "Downloading inference model (14GB)"

  # Check git-lfs
  if ! command -v git-lfs &>/dev/null; then
    log_warn "git-lfs not installed, trying huggingface_hub..."

    python3 - <<EOF_PY
from huggingface_hub import snapshot_download

try:
    snapshot_download(
        repo_id="${HF_INFERENCE_REPO}",
        local_dir="${CHECKPOINT_DIR}/llava-v1.5-7b-M2F2-Det",
        local_dir_use_symlinks=False,
        resume_download=True
    )
    print("✓ Inference model downloaded")
except Exception as e:
    print(f"❌ Download failed: {e}")
    exit(1)
EOF_PY
  else
    cd "$CHECKPOINT_DIR"
    git lfs clone "https://huggingface.co/${HF_INFERENCE_REPO}"
    cd - > /dev/null
  fi

  log_ok "Inference model: ${CHECKPOINT_DIR}/llava-v1.5-7b-M2F2-Det/"
}

# Download CLIP vision encoder (Google Drive)
download_clip_encoder() {
  if [ "$DOWNLOAD_CLIP_ENCODER" = false ]; then
    log_info "Skipping CLIP encoder"
    return
  fi

  log_step "Downloading CLIP vision encoder (400MB)"

  # Check gdown
  if ! command -v gdown &>/dev/null; then
    log_info "Installing gdown..."
    pip install -q gdown
  fi

  gdown "https://drive.google.com/uc?id=${GDRIVE_CLIP_ENCODER_ID}" \
    -O "${WEIGHTS_DIR}/vision_tower.pth"

  log_ok "CLIP encoder: ${WEIGHTS_DIR}/vision_tower.pth"
}

# Download DDVQA dataset
download_ddvqa_dataset() {
  if [ "$DOWNLOAD_DDVQA_DATASET" = false ]; then
    log_info "Skipping DDVQA dataset"
    return
  fi

  log_step "Preparing DDVQA dataset"

  # Check if already unpacked
  if [ -d "${DATASET_DIR}/c40/train" ] && [ -d "${DATASET_DIR}/c40/test" ]; then
    log_ok "DDVQA c40 dataset already present"
    return
  fi

  # Option 1: unzip local zip file
  if [ -f "${DDVQA_LOCAL_ZIP}" ]; then
    log_info "Found local c40.zip, extracting..."
    unzip -q "${DDVQA_LOCAL_ZIP}" -d "${DATASET_DIR}/"
    log_ok "DDVQA dataset: ${DATASET_DIR}/c40/"
    return
  fi

  # Option 2: download from Google Drive
  log_info "Downloading DDVQA dataset from Google Drive..."

  # Check and install gdown
  if ! command -v gdown &>/dev/null; then
    log_info "Installing gdown..."
    pip install -q gdown
  fi

  # Create target directory
  mkdir -p "$(dirname "${DDVQA_LOCAL_ZIP}")"

  # Download c40.zip
  log_info "Downloading... (this may take a few minutes)"
  gdown --fuzzy "${DDVQA_GDRIVE_URL}" -O "${DDVQA_LOCAL_ZIP}"

  # Extract
  if [ -f "${DDVQA_LOCAL_ZIP}" ]; then
    log_info "Extracting c40.zip..."
    unzip -q "${DDVQA_LOCAL_ZIP}" -d "${DATASET_DIR}/"
    log_ok "DDVQA dataset: ${DATASET_DIR}/c40/"
  else
    log_warn "Google Drive download failed"
    log_info "Please download manually: ${DDVQA_GDRIVE_URL}"
    log_info "and place c40.zip at: ${DDVQA_LOCAL_ZIP}"
    return 1
  fi
}

# ============================================================
# Verify downloads
# ============================================================
verify_downloads() {
  log_step "Verifying downloaded files"

  local all_ok=true

  # Stage-1
  if [ "$DOWNLOAD_STAGE1_WEIGHTS" = true ]; then
    if [ -f "${WEIGHTS_DIR}/M2F2_Det_densenet121.pth" ]; then
      log_ok "Stage-1 detector"
    else
      log_warn "Missing: Stage-1 detector"
      all_ok=false
    fi
  fi

  # Stage-2
  if [ "$DOWNLOAD_STAGE2_INIT_WEIGHTS" = true ]; then
    if [ -d "${CHECKPOINT_DIR}/llava-1.5-7b-deepfake-rand-proj-v1" ]; then
      log_ok "Stage-2 init weights"
    else
      log_warn "Missing: Stage-2 init weights"
      all_ok=false
    fi
  fi

  # LLaVA Base
  if [ "$DOWNLOAD_LLAVA_BASE" = true ]; then
    if [ -d "${CHECKPOINT_DIR}/llava-v1.5-7b" ]; then
      log_ok "LLaVA base model"
    else
      log_warn "Missing: LLaVA base model"
      all_ok=false
    fi
  fi

  # Inference model
  if [ "$DOWNLOAD_INFERENCE_MODEL" = true ]; then
    if [ -d "${CHECKPOINT_DIR}/llava-v1.5-7b-M2F2-Det" ]; then
      log_ok "Inference model"
    else
      log_warn "Missing: inference model"
      all_ok=false
    fi
  fi

  # DDVQA dataset
  if [ "$DOWNLOAD_DDVQA_DATASET" = true ]; then
    if [ -d "${DATASET_DIR}/c40" ]; then
      local train_count=$(find "${DATASET_DIR}/c40/train" -type f 2>/dev/null | wc -l)
      log_ok "DDVQA dataset (${train_count} training images)"
    else
      log_warn "Missing: DDVQA dataset"
      all_ok=false
    fi
  fi

  if [ "$all_ok" = true ]; then
    echo -e "\n${BOLD}${GREEN}✅ All files downloaded successfully!${RESET}"
  else
    echo -e "\n${BOLD}${YELLOW}⚠️  Some files failed to download${RESET}"
  fi
}

# ============================================================
# Main flow
# ============================================================
main() {
  echo "========================================"
  echo "M2F2_Det Weights Download Tool"
  echo "========================================"

  check_dependencies
  create_directories

  download_stage1_weights
  download_stage2_init_weights
  download_llava_base
  download_inference_model
  download_clip_encoder
  download_ddvqa_dataset

  verify_downloads

  echo ""
  echo "========================================"
  echo "Next step: bash scripts/verify_env.sh"
  echo "========================================"
}

main
