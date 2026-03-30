#!/bin/bash
# CUDA 12.1 installation script
# Usage: bash setup/install_cuda.sh

set -euo pipefail

# ============================================================
# Colored output helpers
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
# Check whether CUDA 12.1 is already installed
# ============================================================
check_existing_cuda() {
  log_step "Checking existing CUDA installation"

  if [ -d "/usr/local/cuda-12.1" ]; then
    log_ok "CUDA 12.1 is installed at /usr/local/cuda-12.1"

    # Check environment variables
    if [ -n "${CUDA_HOME:-}" ] && [ "$CUDA_HOME" = "/usr/local/cuda-12.1" ]; then
      log_ok "CUDA_HOME is set correctly"
      return 0
    else
      log_warn "CUDA_HOME is not set or incorrect"
      log_info "Run: export CUDA_HOME=/usr/local/cuda-12.1"
      log_info "Run: export PATH=/usr/local/cuda-12.1/bin:\$PATH"
      log_info "Run: export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:\$LD_LIBRARY_PATH"
      return 1
    fi
  fi

  log_warn "CUDA 12.1 not found"
  return 1
}

# ============================================================
# Install CUDA 12.1
# ============================================================
install_cuda_12_1() {
  log_step "Installing CUDA 12.1"

  # Check system architecture
  ARCH=$(uname -m)
  if [ "$ARCH" != "x86_64" ]; then
    log_err "Unsupported architecture: $ARCH (x86_64 only)"
    exit 1
  fi

  # Check Ubuntu version
  if [ -f "/etc/os-release" ]; then
    . /etc/os-release
    log_info "Detected system: $NAME $VERSION_ID"

    case "$VERSION_ID" in
      "20.04")
        CUDA_REPO="ubuntu2004"
        ;;
      "22.04")
        CUDA_REPO="ubuntu2204"
        ;;
      *)
        log_warn "Ubuntu version not explicitly supported: $VERSION_ID"
        log_info "Trying ubuntu2204 repo"
        CUDA_REPO="ubuntu2204"
        ;;
    esac
  else
    log_err "Unable to detect system version"
    exit 1
  fi

  # Download and install CUDA keyring
  log_info "Adding CUDA repo keyring..."

  cd /tmp
  wget -q https://developer.download.nvidia.com/compute/cuda/repos/${CUDA_REPO}/x86_64/cuda-keyring_1.1-1_all.deb

  if [ ! -f "cuda-keyring_1.1-1_all.deb" ]; then
    log_err "Failed to download CUDA keyring"
    exit 1
  fi

  sudo dpkg -i cuda-keyring_1.1-1_all.deb
  log_ok "CUDA keyring installed"

  # Update apt cache
  log_info "Updating apt cache..."
  sudo apt-get update -qq

  # Install CUDA 12.1
  log_info "Installing CUDA 12.1 toolkit (this may take a few minutes)..."
  sudo apt-get install -y cuda-toolkit-12-1

  log_ok "CUDA 12.1 installation complete"

  # Clean up temp files
  rm -f /tmp/cuda-keyring_1.1-1_all.deb
}

# ============================================================
# Configure environment variables
# ============================================================
setup_environment() {
  log_step "Configuring CUDA environment variables"

  # Apply to current shell
  export CUDA_HOME=/usr/local/cuda-12.1
  export PATH=/usr/local/cuda-12.1/bin:$PATH
  export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:${LD_LIBRARY_PATH:-}

  log_ok "Environment variables set (current shell)"

  # Add to ~/.bashrc (if missing)
  BASHRC="$HOME/.bashrc"

  if ! grep -q "CUDA_HOME=/usr/local/cuda-12.1" "$BASHRC" 2>/dev/null; then
    log_info "Adding CUDA environment variables to ~/.bashrc"

    cat >> "$BASHRC" <<'EOF_INNER'

# CUDA 12.1 Environment Variables
export CUDA_HOME=/usr/local/cuda-12.1
export PATH=/usr/local/cuda-12.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH
EOF_INNER

    log_ok "Added to ~/.bashrc (takes effect next login)"
  else
    log_ok "CUDA configuration already present in ~/.bashrc"
  fi
}

# ============================================================
# Verify installation
# ============================================================
verify_installation() {
  log_step "Verifying CUDA installation"

  # Check nvcc
  if command -v nvcc &>/dev/null; then
    NVCC_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | tr -d ',')
    log_ok "nvcc version: $NVCC_VERSION"
  else
    log_err "nvcc not found"
    return 1
  fi

  # Check CUDA directory
  if [ -d "/usr/local/cuda-12.1" ]; then
    log_ok "CUDA 12.1 directory: /usr/local/cuda-12.1"
  else
    log_err "CUDA 12.1 directory not found"
    return 1
  fi

  # Check environment variables
  if [ -n "${CUDA_HOME:-}" ]; then
    log_ok "CUDA_HOME: $CUDA_HOME"
  else
    log_warn "CUDA_HOME is not set"
  fi

  echo ""
  log_ok "CUDA 12.1 verification succeeded!"
}

# ============================================================
# Install Flash Attention
# ============================================================
install_flash_attention() {
  log_step "Installing Flash Attention 2.5.7"

  if ! command -v pip &>/dev/null; then
    log_err "pip not found, install Python first"
    return 1
  fi

  log_info "Building Flash Attention with CUDA 12.1..."
  log_info "This may take 5-10 minutes..."

  # Ensure environment variables are set
  export CUDA_HOME=/usr/local/cuda-12.1
  export PATH=/usr/local/cuda-12.1/bin:$PATH
  export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:${LD_LIBRARY_PATH:-}

  # Install flash-attn
  pip install flash-attn==2.5.7 --no-build-isolation

  if [ $? -eq 0 ]; then
    log_ok "Flash Attention 2.5.7 installed"
  else
    log_err "Flash Attention installation failed"
    log_info "Possible dependency: sudo apt-get install -y ninja-build"
    return 1
  fi
}

# ============================================================
# Main flow
# ============================================================
main() {
  echo "========================================"
  echo "CUDA 12.1 Installation Tool"
  echo "========================================"

  # Check if already installed
  if check_existing_cuda; then
    log_info "CUDA 12.1 is installed and configured"

    read -p "Reconfigure environment variables? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
      setup_environment
    fi

    read -p "Install Flash Attention? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
      install_flash_attention
    fi

    exit 0
  fi

  # Install CUDA
  install_cuda_12_1

  # Configure environment
  setup_environment

  # Verify installation
  verify_installation

  # Ask whether to install Flash Attention
  echo ""
  read -p "Install Flash Attention 2.5.7 now? (y/N): " -n 1 -r
  echo
  if [[ $REPLY =~ ^[Yy]$ ]]; then
    install_flash_attention
  else
    log_info "You can install Flash Attention later with:"
    log_info "  export CUDA_HOME=/usr/local/cuda-12.1"
    log_info "  export PATH=/usr/local/cuda-12.1/bin:\$PATH"
    log_info "  pip install flash-attn==2.5.7 --no-build-isolation"
  fi

  echo ""
  echo "========================================"
  echo "Installation complete!"
  echo "========================================"
  echo ""
  echo "Important notes:"
  echo "  1. Log out/in or run: source ~/.bashrc"
  echo "  2. Verify installation: nvcc --version"
  echo "  3. If using venv, activate it before installing flash-attn"
  echo ""
}

main "$@"
