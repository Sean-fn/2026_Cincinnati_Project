#!/bin/bash
# CUDA 11.7 安装脚本
# 用法: bash setup/install_cuda.sh

set -euo pipefail

# ============================================================
# 彩色输出函数
# ============================================================
if [ -t 1 ]; then
  BOLD="$(tput bold)"; RESET="$(tput sgr0)"
  GREEN="$(tput setaf 2)"; YELLOW="$(tput setaf 3)"
  BLUE="$(tput setaf 4)"; CYAN="$(tput setaf 6)"
  RED="$(tput setaf 1)"
else
  BOLD=""; RESET=""; GREEN=""; YELLOW=""; BLUE=""; CYAN=""; RED=""
fi

log_step() { echo -e "\n${BOLD}${BLUE}==>${RESET} ${BOLD}$*${RESET}"; }
log_info() { echo -e "${CYAN}[INFO]${RESET} $*"; }
log_ok()   { echo -e "${GREEN}[ OK ]${RESET} $*"; }
log_warn() { echo -e "${YELLOW}[WARN]${RESET} $*"; }
log_err()  { echo -e "${RED}[ERR ]${RESET} $*"; }

# ============================================================
# 检查是否已安装 CUDA 11.7
# ============================================================
check_existing_cuda() {
  log_step "检查现有 CUDA 安装"

  if [ -d "/usr/local/cuda-11.7" ]; then
    log_ok "CUDA 11.7 已安装在 /usr/local/cuda-11.7"

    # 检查环境变量
    if [ -n "${CUDA_HOME:-}" ] && [ "$CUDA_HOME" = "/usr/local/cuda-11.7" ]; then
      log_ok "CUDA_HOME 已正确设置"
      return 0
    else
      log_warn "CUDA_HOME 未设置或不正确"
      log_info "请运行: export CUDA_HOME=/usr/local/cuda-11.7"
      log_info "请运行: export PATH=/usr/local/cuda-11.7/bin:\$PATH"
      log_info "请运行: export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64:\$LD_LIBRARY_PATH"
      return 1
    fi
  fi

  log_warn "未找到 CUDA 11.7 安装"
  return 1
}

# ============================================================
# 安装 CUDA 11.7
# ============================================================
install_cuda_11_7() {
  log_step "安装 CUDA 11.7"

  # 检查系统架构
  ARCH=$(uname -m)
  if [ "$ARCH" != "x86_64" ]; then
    log_err "不支持的架构: $ARCH (仅支持 x86_64)"
    exit 1
  fi

  # 检查 Ubuntu 版本
  if [ -f "/etc/os-release" ]; then
    . /etc/os-release
    log_info "检测到系统: $NAME $VERSION_ID"

    case "$VERSION_ID" in
      "20.04")
        CUDA_REPO="ubuntu2004"
        ;;
      "22.04")
        CUDA_REPO="ubuntu2204"
        ;;
      *)
        log_warn "未明确支持的 Ubuntu 版本: $VERSION_ID"
        log_info "尝试使用 ubuntu2204 仓库"
        CUDA_REPO="ubuntu2204"
        ;;
    esac
  else
    log_err "无法检测系统版本"
    exit 1
  fi

  # 下载并安装 CUDA Keyring
  log_info "添加 CUDA 仓库 keyring..."

  cd /tmp
  wget -q https://developer.download.nvidia.com/compute/cuda/repos/${CUDA_REPO}/x86_64/cuda-keyring_1.1-1_all.deb

  if [ ! -f "cuda-keyring_1.1-1_all.deb" ]; then
    log_err "下载 CUDA keyring 失败"
    exit 1
  fi

  sudo dpkg -i cuda-keyring_1.1-1_all.deb
  log_ok "CUDA keyring 安装完成"

  # 更新 apt 缓存
  log_info "更新 apt 缓存..."
  sudo apt-get update -qq

  # 安装 CUDA 11.7
  log_info "安装 CUDA 11.7 toolkit (这可能需要几分钟)..."
  sudo apt-get install -y cuda-toolkit-11-7

  log_ok "CUDA 11.7 安装完成"

  # 清理临时文件
  rm -f /tmp/cuda-keyring_1.1-1_all.deb
}

# ============================================================
# 配置环境变量
# ============================================================
setup_environment() {
  log_step "配置 CUDA 环境变量"

  # 添加到当前 shell
  export CUDA_HOME=/usr/local/cuda-11.7
  export PATH=/usr/local/cuda-11.7/bin:$PATH
  export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64:${LD_LIBRARY_PATH:-}

  log_ok "环境变量已设置 (当前 shell)"

  # 添加到 ~/.bashrc (如果不存在)
  BASHRC="$HOME/.bashrc"

  if ! grep -q "CUDA_HOME=/usr/local/cuda-11.7" "$BASHRC" 2>/dev/null; then
    log_info "添加 CUDA 环境变量到 ~/.bashrc"

    cat >> "$BASHRC" <<'EOF'

# CUDA 11.7 Environment Variables
export CUDA_HOME=/usr/local/cuda-11.7
export PATH=/usr/local/cuda-11.7/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH
EOF

    log_ok "已添加到 ~/.bashrc (下次登录生效)"
  else
    log_ok "~/.bashrc 中已存在 CUDA 配置"
  fi
}

# ============================================================
# 验证安装
# ============================================================
verify_installation() {
  log_step "验证 CUDA 安装"

  # 检查 nvcc
  if command -v nvcc &>/dev/null; then
    NVCC_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | tr -d ',')
    log_ok "nvcc 版本: $NVCC_VERSION"
  else
    log_err "nvcc 未找到"
    return 1
  fi

  # 检查 CUDA 目录
  if [ -d "/usr/local/cuda-11.7" ]; then
    log_ok "CUDA 11.7 目录: /usr/local/cuda-11.7"
  else
    log_err "CUDA 11.7 目录不存在"
    return 1
  fi

  # 检查环境变量
  if [ -n "${CUDA_HOME:-}" ]; then
    log_ok "CUDA_HOME: $CUDA_HOME"
  else
    log_warn "CUDA_HOME 未设置"
  fi

  echo ""
  log_ok "CUDA 11.7 安装验证成功!"
}

# ============================================================
# 安装 Flash Attention
# ============================================================
install_flash_attention() {
  log_step "安装 Flash Attention 2.5.7"

  if ! command -v pip &>/dev/null; then
    log_err "pip 未找到,请先安装 Python 环境"
    return 1
  fi

  log_info "使用 CUDA 11.7 编译 Flash Attention..."
  log_info "这可能需要 5-10 分钟..."

  # 确保环境变量设置
  export CUDA_HOME=/usr/local/cuda-11.7
  export PATH=/usr/local/cuda-11.7/bin:$PATH
  export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64:${LD_LIBRARY_PATH:-}

  # 安装 flash-attn
  pip install flash-attn==2.5.7 --no-build-isolation

  if [ $? -eq 0 ]; then
    log_ok "Flash Attention 2.5.7 安装成功"
  else
    log_err "Flash Attention 安装失败"
    log_info "可能需要的依赖: sudo apt-get install -y ninja-build"
    return 1
  fi
}

# ============================================================
# 主流程
# ============================================================
main() {
  echo "========================================"
  echo "CUDA 11.7 安装工具"
  echo "========================================"

  # 检查是否已安装
  if check_existing_cuda; then
    log_info "CUDA 11.7 已安装且配置正确"

    read -p "是否重新配置环境变量? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
      setup_environment
    fi

    read -p "是否安装 Flash Attention? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
      install_flash_attention
    fi

    exit 0
  fi

  # 安装 CUDA
  install_cuda_11_7

  # 配置环境
  setup_environment

  # 验证安装
  verify_installation

  # 询问是否安装 Flash Attention
  echo ""
  read -p "是否现在安装 Flash Attention 2.5.7? (y/N): " -n 1 -r
  echo
  if [[ $REPLY =~ ^[Yy]$ ]]; then
    install_flash_attention
  else
    log_info "稍后可运行以下命令安装 Flash Attention:"
    log_info "  export CUDA_HOME=/usr/local/cuda-11.7"
    log_info "  export PATH=/usr/local/cuda-11.7/bin:\$PATH"
    log_info "  pip install flash-attn==2.5.7 --no-build-isolation"
  fi

  echo ""
  echo "========================================"
  echo "安装完成!"
  echo "========================================"
  echo ""
  echo "重要提示:"
  echo "  1. 请重新登录或运行: source ~/.bashrc"
  echo "  2. 验证安装: nvcc --version"
  echo "  3. 如果使用 venv, 请激活后再安装 flash-attn"
  echo ""
}

main "$@"
