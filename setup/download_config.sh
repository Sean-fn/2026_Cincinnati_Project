#!/bin/bash
# M2F2_Det weights download configuration
# Usage: source setup/download_config.sh

# ============================================================
# Hugging Face repo configuration
# ============================================================
# Training weights repo (includes Stage-1 and Stage-2 init weights)
HF_TRAINING_REPO="Sean-fn/M2F2-Det-Weights"

# Inference model repo (includes final M2F2-Det model)
HF_INFERENCE_REPO="CHELSEA234/llava-v1.5-7b-M2F2-Det"

# LLaVA base model repo
HF_LLAVA_BASE_REPO="liuhaotian/llava-v1.5-7b"

# ============================================================
# Download toggles (set to true to enable)
# ============================================================
DOWNLOAD_STAGE1_WEIGHTS=true          # Stage-1 detector (1.7GB)
DOWNLOAD_STAGE2_INIT_WEIGHTS=true     # Stage-2 init weights (14GB)
DOWNLOAD_LLAVA_BASE=false             # LLaVA base model (13GB, optional)
DOWNLOAD_INFERENCE_MODEL=false        # Inference model (14GB, inference only)
DOWNLOAD_CLIP_ENCODER=false           # CLIP vision encoder (400MB, optional)
DOWNLOAD_DDVQA_DATASET=true           # DDVQA dataset (unzip local c40.zip or download from Google Drive)

# ============================================================
# Local storage paths
# ============================================================
CHECKPOINT_DIR="./checkpoints"
WEIGHTS_DIR="./utils/weights"
DATASET_DIR="./utils/DDVQA_images"

# ============================================================
# External data sources
# ============================================================
# Google Drive file IDs
GDRIVE_CLIP_ENCODER_ID="19oEpKB96xJVSrwkLV0ewje-W2dfBAR58"
GDRIVE_FF_TEST_ID="1tQ0ZwsXXX-K9aWYhn_ELLgViP-T4MC70"

# DDVQA dataset Google Drive
DDVQA_GDRIVE_ID="1rtKKo-bURNlNR7bHzJrGw0V-Kt9Jgu0W"
DDVQA_GDRIVE_URL="https://drive.google.com/file/d/1rtKKo-bURNlNR7bHzJrGw0V-Kt9Jgu0W/view?usp=drive_link"
DDVQA_LOCAL_ZIP="utils/DDVQA_images/c40.zip"
