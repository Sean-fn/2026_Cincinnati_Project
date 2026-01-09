#!/bin/bash
# M2F2_Det 权重下载配置文件
# 用法: source setup/download_config.sh

# ============================================================
# Hugging Face 仓库配置
# ============================================================
# 训练权重仓库 (包含 Stage-1 和 Stage-2 初始化权重)
HF_TRAINING_REPO="Sean-fn/M2F2-Det-Weights"

# 推理模型仓库 (包含最终的 M2F2-Det 模型)
HF_INFERENCE_REPO="CHELSEA234/llava-v1.5-7b-M2F2-Det"

# LLaVA 基础模型仓库
HF_LLAVA_BASE_REPO="liuhaotian/llava-v1.5-7b"

# ============================================================
# 下载开关 (设为 true 启用下载)
# ============================================================
DOWNLOAD_STAGE1_WEIGHTS=true          # Stage-1 检测器 (1.7GB)
DOWNLOAD_STAGE2_INIT_WEIGHTS=true     # Stage-2 初始化权重 (14GB)
DOWNLOAD_LLAVA_BASE=false             # LLaVA 基础模型 (13GB, 可选)
DOWNLOAD_INFERENCE_MODEL=false        # 推理模型 (14GB, 仅推理用)
DOWNLOAD_CLIP_ENCODER=false           # CLIP 视觉编码器 (400MB, 可选)
DOWNLOAD_DDVQA_DATASET=true           # DDVQA 数据集 (解压本地c40.zip或从Google Drive下载)

# ============================================================
# 本地存储路径
# ============================================================
CHECKPOINT_DIR="./checkpoints"
WEIGHTS_DIR="./utils/weights"
DATASET_DIR="./utils/DDVQA_images"

# ============================================================
# 外部数据源
# ============================================================
# Google Drive 文件ID
GDRIVE_CLIP_ENCODER_ID="19oEpKB96xJVSrwkLV0ewje-W2dfBAR58"
GDRIVE_FF_TEST_ID="1tQ0ZwsXXX-K9aWYhn_ELLgViP-T4MC70"

# DDVQA 数据集 Google Drive
DDVQA_GDRIVE_ID="1rtKKo-bURNlNR7bHzJrGw0V-Kt9Jgu0W"
DDVQA_GDRIVE_URL="https://drive.google.com/file/d/1rtKKo-bURNlNR7bHzJrGw0V-Kt9Jgu0W/view?usp=drive_link"
DDVQA_LOCAL_ZIP="utils/DDVQA_images/c40.zip"
