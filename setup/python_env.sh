#!/bin/bash
# Python 环境配置脚本
# 用法: bash setup/python_env.sh [venv|conda]

set -euo pipefail

ENV_TYPE="${1:-venv}"  # 默认使用 venv

log_step() { echo -e "\n==> $*"; }
log_ok()   { echo -e "[OK] $*"; }

case $ENV_TYPE in
  venv)
    log_step "创建 Python venv 环境"
    python3 -m venv venv
    source venv/bin/activate

    log_step "升级 pip"
    pip install --upgrade pip

    log_step "安装依赖 (requirements.txt)"
    pip install -r requirements.txt

    log_ok "venv 环境配置完成"
    echo "激活环境: source venv/bin/activate"
    ;;

  conda)
    log_step "创建 Conda 环境"

    if ! command -v conda &>/dev/null; then
      echo "错误: 未找到 conda"
      exit 1
    fi

    conda env create -f environment.yml

    log_ok "Conda 环境配置完成"
    echo "激活环境: conda activate M2F2_det"
    ;;

  *)
    echo "错误: 未知的环境类型: $ENV_TYPE"
    echo "用法: $0 [venv|conda]"
    exit 1
    ;;
esac
