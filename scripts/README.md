# Scripts Directory Structure

This directory contains all training, inference, and utility scripts for M2F2-Det, organized by functionality.

## Directory Organization

```
scripts/
├── kan/                        # KAN projector experiments
│   ├── stage_2_3_combined.sh  # Combined KAN + QLoRA training
│   ├── stage_2_kan_only.sh    # Train only KAN projector (no quantization)
│   ├── stage_3_qlora_only.sh  # Train only QLoRA (load Stage 2)
│   ├── inference_det.sh       # KAN model detection inference
│   ├── inference_exp.sh       # KAN model explanation inference
│   └── eval.sh                # Complete KAN evaluation pipeline
│
├── original/                   # Original M2F2-Det training
│   ├── finetune_stage_2.sh    # Stage 2: Multi-modal alignment
│   └── finetune_stage_3.sh    # Stage 3: LoRA fine-tuning
│
├── eval/                       # Evaluation & comparison
│   ├── eval_original.sh       # Evaluate original M2F2-Det
│   └── compare_results.sh     # Compare KAN vs original
│
├── utils/                      # Utility scripts
│   ├── merge_lora_weights_deepfake.py          # Merge LoRA/delta weights
│   ├── merge_lora_weights_deepfake_random.py   # Initialize random MLP
│   ├── verify_env.sh                           # Quick environment check
│   └── verify_local.sh                         # Local deployment validation
│
├── config/                     # DeepSpeed configurations
│   ├── zero2.json             # Zero-2 optimization
│   ├── zero3.json             # Zero-3 optimization
│   └── zero3_offload.json     # Zero-3 with CPU offload
│
└── huggingface/               # Model hub integration
    ├── download_from_huggingface.sh  # Download weights
    └── upload_to_huggingface.sh      # Upload weights
```

## Quick Start

### Original M2F2-Det Training

```bash
# Stage 2: Multi-modal alignment
bash scripts/original/finetune_stage_2.sh

# Stage 3: LoRA fine-tuning
bash scripts/original/finetune_stage_3.sh
```

### KAN + QLoRA Training (Low VRAM)

```bash
# Combined training (default)
bash scripts/kan/stage_2_3_combined.sh

# Separate training (more control)
bash scripts/kan/stage_2_kan_only.sh    # Train KAN first
bash scripts/kan/stage_3_qlora_only.sh  # Then train QLoRA
```

### Evaluation

```bash
# Evaluate KAN model
bash scripts/kan/eval.sh

# Evaluate original model
bash scripts/eval/eval_original.sh

# Compare both
bash scripts/eval/compare_results.sh
```

## Configuration Files

### DeepSpeed Configs (`config/`)

- **zero2.json**: Standard multi-GPU training (recommended for most cases)
- **zero3.json**: Advanced memory optimization for very large models
- **zero3_offload.json**: CPU offloading for extreme memory constraints

Reference in training scripts:
```bash
deepspeed --deepspeed ./scripts/config/zero2.json ...
```

## Utility Scripts

### Weight Merging

```bash
# Merge LoRA delta weights with base model
python scripts/utils/merge_lora_weights_deepfake.py \
  --model-base ./base_model \
  --model-path ./delta_weights \
  --save-model-path ./merged_model

# Initialize random MLP for Stage 2
python scripts/utils/merge_lora_weights_deepfake_random.py \
  --model-path ./llava-v1.5-7b \
  --save-model-path ./llava-1.5-7b-deepfake-rand-proj-v1
```

### Environment Verification

```bash
# Quick environment check
bash scripts/utils/verify_env.sh

# Local deployment validation (runs 1-step training)
bash scripts/utils/verify_local.sh
```

## HuggingFace Integration

```bash
# Upload model to HuggingFace Hub
bash scripts/huggingface/upload_to_huggingface.sh YOUR_USERNAME

# Download model from HuggingFace Hub
bash scripts/huggingface/download_from_huggingface.sh YOUR_USERNAME
```

## Migration Notes

If you're updating from older versions:

| Old Path | New Path |
|----------|----------|
| `scripts/finetune_kan_qlora.sh` | `scripts/kan/stage_2_3_combined.sh` |
| `scripts/finetune_stage_2.sh` | `scripts/original/finetune_stage_2.sh` |
| `scripts/finetune_stage_3.sh` | `scripts/original/finetune_stage_3.sh` |
| `scripts/merge_lora_weights_deepfake.py` | `scripts/utils/merge_lora_weights_deepfake.py` |
| `scripts/zero2.json` | `scripts/config/zero2.json` |
| `scripts/eval_original.sh` | `scripts/eval/eval_original.sh` |
| `scripts/download_from_huggingface.sh` | `scripts/huggingface/download_from_huggingface.sh` |

All references in root scripts and documentation have been updated automatically.
