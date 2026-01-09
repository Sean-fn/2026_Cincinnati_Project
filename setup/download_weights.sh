#!/bin/bash
# M2F2_Det 权重下载主脚本
# 用法: bash setup/download_weights.sh [--config FILE] [--quiet]

set -euo pipefail

# ============================================================
# 参数解析
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
# 加载配置
# ============================================================
if [ ! -f "$CONFIG_FILE" ]; then
  echo "错误: 配置文件不存在: $CONFIG_FILE"
  exit 1
fi

source "$CONFIG_FILE"

# ============================================================
# 彩色输出函数
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
# 下载函数
# ============================================================

# 检查依赖
check_dependencies() {
  log_step "检查下载依赖"

  if ! command -v python3 &>/dev/null; then
    echo "错误: 未找到 python3"
    exit 1
  fi

  # 安装 huggingface_hub
  if ! python3 -c "import huggingface_hub" &>/dev/null 2>&1; then
    log_info "安装 huggingface_hub..."
    pip install -q huggingface_hub
  fi

  log_ok "依赖检查完成"
}

# 创建目录结构
create_directories() {
  log_step "创建目录结构"
  mkdir -p "$CHECKPOINT_DIR"
  mkdir -p "$WEIGHTS_DIR"
  mkdir -p "$DATASET_DIR"
  log_ok "目录创建完成"
}

# 下载 Stage-1 检测器权重
download_stage1_weights() {
  if [ "$DOWNLOAD_STAGE1_WEIGHTS" = false ]; then
    log_info "跳过 Stage-1 权重下载"
    return
  fi

  log_step "下载 Stage-1 检测器权重 (1.7GB)"

  python3 - <<EOF
from huggingface_hub import hf_hub_download
import os

try:
    hf_hub_download(
        repo_id="${HF_TRAINING_REPO}",
        filename="weights/M2F2_Det_densenet121.pth",
        local_dir="${WEIGHTS_DIR%/*}",
        local_dir_use_symlinks=False
    )
    print("✓ Stage-1 权重下载完成")
except Exception as e:
    print(f"❌ 下载失败: {e}")
    exit(1)
EOF

  log_ok "Stage-1 权重: ${WEIGHTS_DIR}/M2F2_Det_densenet121.pth"
}

# 下载 Stage-2 初始化权重
download_stage2_init_weights() {
  if [ "$DOWNLOAD_STAGE2_INIT_WEIGHTS" = false ]; then
    log_info "跳过 Stage-2 初始化权重下载"
    return
  fi

  log_step "下载 Stage-2 初始化权重 (14GB, 可能需要较长时间)"

  python3 - <<EOF
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
    print("✓ Stage-2 初始化权重下载完成")
except Exception as e:
    print(f"❌ 下载失败: {e}")
    exit(1)
EOF

  log_ok "Stage-2 权重: ${CHECKPOINT_DIR}/llava-1.5-7b-deepfake-rand-proj-v1/"
}

# 下载 LLaVA 基础模型
download_llava_base() {
  if [ "$DOWNLOAD_LLAVA_BASE" = false ]; then
    log_info "跳过 LLaVA 基础模型下载"
    return
  fi

  log_step "下载 LLaVA-1.5-7b 基础模型 (13GB)"

  python3 - <<EOF
from huggingface_hub import snapshot_download

try:
    snapshot_download(
        repo_id="${HF_LLAVA_BASE_REPO}",
        local_dir="${CHECKPOINT_DIR}/llava-v1.5-7b",
        local_dir_use_symlinks=False,
        resume_download=True
    )
    print("✓ LLaVA 基础模型下载完成")
except Exception as e:
    print(f"❌ 下载失败: {e}")
    exit(1)
EOF

  log_ok "LLaVA 基础模型: ${CHECKPOINT_DIR}/llava-v1.5-7b/"
}

# 下载推理模型
download_inference_model() {
  if [ "$DOWNLOAD_INFERENCE_MODEL" = false ]; then
    log_info "跳过推理模型下载"
    return
  fi

  log_step "下载推理模型 (14GB)"

  # 检查 git lfs
  if ! command -v git-lfs &>/dev/null; then
    log_warn "未安装 git-lfs, 尝试使用 huggingface_hub..."

    python3 - <<EOF
from huggingface_hub import snapshot_download

try:
    snapshot_download(
        repo_id="${HF_INFERENCE_REPO}",
        local_dir="${CHECKPOINT_DIR}/llava-v1.5-7b-M2F2-Det",
        local_dir_use_symlinks=False,
        resume_download=True
    )
    print("✓ 推理模型下载完成")
except Exception as e:
    print(f"❌ 下载失败: {e}")
    exit(1)
EOF
  else
    cd "$CHECKPOINT_DIR"
    git lfs clone "https://huggingface.co/${HF_INFERENCE_REPO}"
    cd - > /dev/null
  fi

  log_ok "推理模型: ${CHECKPOINT_DIR}/llava-v1.5-7b-M2F2-Det/"
}

# 下载 CLIP 视觉编码器 (Google Drive)
download_clip_encoder() {
  if [ "$DOWNLOAD_CLIP_ENCODER" = false ]; then
    log_info "跳过 CLIP 编码器下载"
    return
  fi

  log_step "下载 CLIP 视觉编码器 (400MB)"

  # 检查 gdown
  if ! command -v gdown &>/dev/null; then
    log_info "安装 gdown..."
    pip install -q gdown
  fi

  gdown "https://drive.google.com/uc?id=${GDRIVE_CLIP_ENCODER_ID}" \
    -O "${WEIGHTS_DIR}/vision_tower.pth"

  log_ok "CLIP 编码器: ${WEIGHTS_DIR}/vision_tower.pth"
}

# 下载 DDVQA 数据集
download_ddvqa_dataset() {
  if [ "$DOWNLOAD_DDVQA_DATASET" = false ]; then
    log_info "跳过 DDVQA 数据集下载"
    return
  fi

  log_step "处理 DDVQA 数据集"

  # 检查是否已解压
  if [ -d "${DATASET_DIR}/c40/train" ] && [ -d "${DATASET_DIR}/c40/test" ]; then
    log_ok "DDVQA c40 数据集已存在"
    return
  fi

  # 方案1: 解压本地 zip 文件
  if [ -f "${DDVQA_LOCAL_ZIP}" ]; then
    log_info "发现本地 c40.zip, 正在解压..."
    unzip -q "${DDVQA_LOCAL_ZIP}" -d "${DATASET_DIR}/"
    log_ok "DDVQA 数据集: ${DATASET_DIR}/c40/"
    return
  fi

  # 方案2: 从 Google Drive 下载
  log_info "从 Google Drive 下载 DDVQA 数据集..."

  # 检查并安装 gdown
  if ! command -v gdown &>/dev/null; then
    log_info "安装 gdown..."
    pip install -q gdown
  fi

  # 创建目标目录
  mkdir -p "$(dirname "${DDVQA_LOCAL_ZIP}")"

  # 下载 c40.zip
  log_info "下载中... (这可能需要几分钟)"
  gdown --fuzzy "${DDVQA_GDRIVE_URL}" -O "${DDVQA_LOCAL_ZIP}"

  # 解压
  if [ -f "${DDVQA_LOCAL_ZIP}" ]; then
    log_info "正在解压 c40.zip..."
    unzip -q "${DDVQA_LOCAL_ZIP}" -d "${DATASET_DIR}/"
    log_ok "DDVQA 数据集: ${DATASET_DIR}/c40/"
  else
    log_warn "Google Drive 下载失败"
    log_info "请手动下载: ${DDVQA_GDRIVE_URL}"
    log_info "并将 c40.zip 放置在: ${DDVQA_LOCAL_ZIP}"
    return 1
  fi
}

# ============================================================
# 验证下载
# ============================================================
verify_downloads() {
  log_step "验证下载文件"

  local all_ok=true

  # Stage-1
  if [ "$DOWNLOAD_STAGE1_WEIGHTS" = true ]; then
    if [ -f "${WEIGHTS_DIR}/M2F2_Det_densenet121.pth" ]; then
      log_ok "Stage-1 检测器"
    else
      log_warn "缺失: Stage-1 检测器"
      all_ok=false
    fi
  fi

  # Stage-2
  if [ "$DOWNLOAD_STAGE2_INIT_WEIGHTS" = true ]; then
    if [ -d "${CHECKPOINT_DIR}/llava-1.5-7b-deepfake-rand-proj-v1" ]; then
      log_ok "Stage-2 初始化权重"
    else
      log_warn "缺失: Stage-2 初始化权重"
      all_ok=false
    fi
  fi

  # LLaVA Base
  if [ "$DOWNLOAD_LLAVA_BASE" = true ]; then
    if [ -d "${CHECKPOINT_DIR}/llava-v1.5-7b" ]; then
      log_ok "LLaVA 基础模型"
    else
      log_warn "缺失: LLaVA 基础模型"
      all_ok=false
    fi
  fi

  # 推理模型
  if [ "$DOWNLOAD_INFERENCE_MODEL" = true ]; then
    if [ -d "${CHECKPOINT_DIR}/llava-v1.5-7b-M2F2-Det" ]; then
      log_ok "推理模型"
    else
      log_warn "缺失: 推理模型"
      all_ok=false
    fi
  fi

  # DDVQA 数据集
  if [ "$DOWNLOAD_DDVQA_DATASET" = true ]; then
    if [ -d "${DATASET_DIR}/c40" ]; then
      local train_count=$(find "${DATASET_DIR}/c40/train" -type f 2>/dev/null | wc -l)
      log_ok "DDVQA 数据集 (${train_count} 训练图片)"
    else
      log_warn "缺失: DDVQA 数据集"
      all_ok=false
    fi
  fi

  if [ "$all_ok" = true ]; then
    echo -e "\n${BOLD}${GREEN}✅ 所有文件下载完成!${RESET}"
  else
    echo -e "\n${BOLD}${YELLOW}⚠️  部分文件下载失败${RESET}"
  fi
}

# ============================================================
# 主流程
# ============================================================
main() {
  echo "========================================"
  echo "M2F2_Det 权重下载工具"
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
  echo "下一步: bash scripts/verify_env.sh"
  echo "========================================"
}

main
