#!/bin/bash
# ============================================================
# M2F2_Det 项目环境设置脚本
#
# 功能:
#   1. 创建 Python 虚拟环境 (venv 或 conda)
#   2. 安装 Python 依赖 (requirements.txt)
#   3. 下载预训练权重 (Stage-1, Stage-2, LLaVA 等)
#   4. 下载 DDVQA 数据集
#   5. 安装 CUDA 11.7 (可选但推荐)
#   6. 验证环境配置
#
# 用法:
#   bash setup/setup_m2f2_project.sh [--env venv|conda] [--skip-cuda] [--skip-download]
# ============================================================

set -euo pipefail

# ============================================================
# 彩色输出函数
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
# 参数解析
# ============================================================
ENV_TYPE="venv"          # 默认使用 venv
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
      echo "用法: $0 [OPTIONS]"
      echo ""
      echo "选项:"
      echo "  --env venv|conda    选择环境类型 (默认: venv)"
      echo "  --skip-cuda         跳过 CUDA 11.7 安装"
      echo "  --skip-download     跳过权重和数据集下载"
      echo "  -h, --help          显示帮助信息"
      echo ""
      echo "示例:"
      echo "  $0                          # 使用默认配置"
      echo "  $0 --env conda              # 使用 conda 环境"
      echo "  $0 --skip-cuda              # 跳过 CUDA 安装"
      echo "  $0 --env venv --skip-download  # 仅设置环境,不下载"
      exit 0
      ;;
    *)
      log_err "未知选项: $1"
      echo "使用 --help 查看帮助"
      exit 1
      ;;
  esac
done

# ============================================================
# 检查脚本位置
# ============================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

log_info "项目根目录: $PROJECT_ROOT"
cd "$PROJECT_ROOT"

# ============================================================
# Banner
# ============================================================
echo ""
echo "========================================"
echo "M2F2_Det 项目环境设置"
echo "========================================"
echo ""
log_info "配置选项:"
echo "  - Python 环境: $ENV_TYPE"
echo "  - 跳过 CUDA: $SKIP_CUDA"
echo "  - 跳过下载: $SKIP_DOWNLOAD"
echo ""

# ============================================================
# 1. 创建 Python 环境
# ============================================================
log_step "1/6 创建 Python 环境 ($ENV_TYPE)"

if [ -f "setup/python_env.sh" ]; then
  run_spin "setup Python environment" bash setup/python_env.sh "$ENV_TYPE"
else
  log_warn "setup/python_env.sh 未找到, 使用默认方式"

  if [ "$ENV_TYPE" = "venv" ]; then
    if [ -d "venv" ]; then
      log_ok "venv 环境已存在"
    else
      run_spin "create venv" python3 -m venv venv
    fi

    log_info "激活 venv..."
    source venv/bin/activate

    run_spin "upgrade pip" pip install --upgrade pip
    run_spin "install requirements" pip install -r requirements.txt
  elif [ "$ENV_TYPE" = "conda" ]; then
    if ! command -v conda &>/dev/null; then
      log_err "conda 未找到,请先安装 Anaconda/Miniconda"
      exit 1
    fi

    run_spin "create conda env" conda env create -f environment.yml
    log_ok "Conda 环境创建完成"
    log_info "请手动激活: conda activate M2F2_det"
  else
    log_err "不支持的环境类型: $ENV_TYPE"
    exit 1
  fi
fi

# ============================================================
# 2. 下载预训练权重和数据集
# ============================================================
if [ "$SKIP_DOWNLOAD" = false ]; then
  log_step "2/6 下载预训练权重和数据集"

  if [ -f "setup/download_weights.sh" ]; then
    bash setup/download_weights.sh
  else
    log_warn "setup/download_weights.sh 未找到"

    if [ -f "scripts/download_from_huggingface.sh" ]; then
      log_info "使用旧版下载脚本"
      run_spin "download weights" bash scripts/download_from_huggingface.sh "Sean-fn"
    else
      log_err "未找到任何下载脚本"
    fi
  fi
else
  log_step "2/6 跳过下载 (--skip-download)"
fi

# ============================================================
# 3. 安装 CUDA 11.7
# ============================================================
if [ "$SKIP_CUDA" = false ]; then
  log_step "3/6 安装 CUDA 11.7"

  if [ -d "/usr/local/cuda-11.7" ]; then
    log_ok "CUDA 11.7 已安装"

    # 检查环境变量
    if [ -n "${CUDA_HOME:-}" ] && [ "$CUDA_HOME" = "/usr/local/cuda-11.7" ]; then
      log_ok "CUDA_HOME 已配置"
    else
      log_warn "CUDA_HOME 未配置,请运行: export CUDA_HOME=/usr/local/cuda-11.7"
    fi
  else
    log_warn "CUDA 11.7 未安装"

    if [ -f "setup/install_cuda.sh" ]; then
      log_info "CUDA 11.7 是训练所必需的 (Flash Attention 要求)"
      bash setup/install_cuda.sh
    else
      log_warn "setup/install_cuda.sh 未找到"
    fi
  fi
else
  log_step "3/6 跳过 CUDA 安装 (--skip-cuda)"
fi

# ============================================================
# 4. 验证环境
# ============================================================
log_step "4/6 验证环境配置"

if [ -f "scripts/verify_env.sh" ]; then
  bash scripts/verify_env.sh
else
  log_warn "scripts/verify_env.sh 未找到,跳过验证"

  # 简单验证
  log_info "基本检查:"

  if command -v python3 &>/dev/null; then
    log_ok "Python: $(python3 --version)"
  else
    log_err "Python3 未找到"
  fi

  if python3 -c "import torch" 2>/dev/null; then
    log_ok "PyTorch 已安装"
  else
    log_warn "PyTorch 未安装或导入失败"
  fi

  if python3 -c "import transformers" 2>/dev/null; then
    log_ok "Transformers 已安装"
  else
    log_warn "Transformers 未安装或导入失败"
  fi
fi

# ============================================================
# 5. 检查关键文件
# ============================================================
log_step "5/6 检查关键文件和目录"

check_path() {
  local path="$1"
  local name="$2"

  if [ -e "$path" ]; then
    log_ok "$name: $path"
  else
    log_warn "$name 缺失: $path"
  fi
}

check_path "checkpoints/llava-1.5-7b-deepfake-rand-proj-v1" "Stage-2 初始化权重"
check_path "utils/weights/M2F2_Det_densenet121.pth" "Stage-1 检测器"
check_path "utils/DDVQA_images/c40" "DDVQA 数据集"
check_path "requirements.txt" "Python 依赖列表"
check_path "environment.yml" "Conda 环境配置"

# ============================================================
# 6. 完成总结
# ============================================================
log_step "6/6 设置完成"

echo ""
echo "========================================"
echo -e "${BOLD}${GREEN}✅ M2F2_Det 项目环境设置完成!${RESET}"
echo "========================================"
echo ""

log_info "环境配置:"
if [ "$ENV_TYPE" = "venv" ]; then
  echo "  - Python 环境: venv"
  echo "  - 激活命令: ${BOLD}source venv/bin/activate${RESET}"
elif [ "$ENV_TYPE" = "conda" ]; then
  echo "  - Python 环境: conda"
  echo "  - 激活命令: ${BOLD}conda activate M2F2_det${RESET}"
fi

echo ""
log_info "训练流程:"
echo "  1) Stage-1 训练: ${BOLD}bash stage_1_train.sh${RESET}"
echo "  2) Stage-2 训练: ${BOLD}bash stage_2_train.sh${RESET}"
echo "  3) Stage-3 训练: ${BOLD}bash stage_3_train.sh${RESET}"

echo ""
log_info "推理流程:"
echo "  1) 检测推理: ${BOLD}bash stage_3_inference_det.sh${RESET}"
echo "  2) 解释生成: ${BOLD}bash stage_3_inference_exp.sh${RESET}"
echo "  3) 评估结果: ${BOLD}python eval/eval_judgement.py${RESET}"

echo ""
log_info "低显存训练 (KAN + QLoRA):"
echo "  - 运行命令: ${BOLD}bash scripts/kan/stage_2_3_combined.sh${RESET}"

echo ""
log_info "文档资源:"
echo "  - 快速开始: ${BOLD}QUICKSTART.md${RESET}"
echo "  - 设置文档: ${BOLD}setup/README.md${RESET}"
echo "  - 项目文档: ${BOLD}CLAUDE.md${RESET}"

if [ "$SKIP_CUDA" = false ] && [ ! -d "/usr/local/cuda-11.7" ]; then
  echo ""
  log_warn "提醒: CUDA 11.7 未安装"
  echo "  - 安装命令: ${BOLD}bash setup/install_cuda.sh${RESET}"
  echo "  - 训练需要 CUDA 11.7 和 Flash Attention"
fi

echo ""
