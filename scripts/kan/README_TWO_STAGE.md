# KAN Two-Stage Training Pipeline

This document describes the two-stage training approach for the KAN-based M2F2-Det model.

## Overview

The training is split into two sequential stages:

1. **Stage 2: KAN Projector Training** - Train only the KAN projector while freezing LLM
2. **Stage 3: LoRA Fine-tuning** - Fine-tune LLM with LoRA while also fine-tuning KAN

This approach follows a **curriculum learning** strategy, allowing the KAN projector to learn robust feature mappings before introducing LLM adaptation.

---

## Stage 2: KAN Projector Training Only

**Script:** `stage_2_kan_only.sh`

### What Gets Trained
- ✅ **KAN Projector** (`deepfake_mlp_adapter`)
  - Maps [P_real, P_fake] → LLM hidden space (4096D)
  - Architecture: 2 → 128 → 4096 with B-spline activation

### What Gets Frozen
- ❄️ **LLM Backbone** (Vicuna-7B)
- ❄️ **Vision Tower** (CLIP ViT-L/14@336px)
- ❄️ **MM Projector** (CLIP features → LLM)
- ❄️ **M2F2-Det Binary Detector**

### Key Parameters
```bash
--lora_enable False                    # No LoRA in Stage 2
--tune_mm_mlp_adapter False            # Freeze MM projector
--tune_deepfake_mlp_adapter True       # Train KAN projector
--freeze_backbone True                 # Freeze LLM
--freeze_mm_mlp_adapter True           # Freeze MM projector

--learning_rate 5e-4                   # Higher LR for projector-only training
--num_train_epochs 3                   # More epochs to converge
--per_device_train_batch_size 8        # Larger batch size (less memory needed)
```

### Expected Training Time
- **~3-4 hours** on RTX 4080 SUPER (16GB)
- VRAM Usage: ~10-12 GB

### Output
- **Checkpoint:** `./checkpoints/llava-v1.5-7b-deepfake-kan-stage2/`
- This checkpoint will be used as the base for Stage 3

---

## Stage 3: LoRA Fine-tuning

**Script:** `stage_3_lora_finetune.sh`

### What Gets Trained
- ✅ **LLM (LoRA adapters)**
  - Rank: 64, Alpha: 128
  - Target modules: q_proj, v_proj
- ✅ **KAN Projector** (fine-tuning)
- ✅ **MM Projector** (fine-tuning)

### What Gets Frozen
- ❄️ **LLM Backbone** (frozen, only LoRA adapters trained)
- ❄️ **Vision Tower** (CLIP)
- ❄️ **M2F2-Det Binary Detector**

### Key Parameters
```bash
--lora_enable True                     # Enable LoRA
--lora_r 64                            # LoRA rank
--lora_alpha 128                       # LoRA alpha
--bits 4                               # 4-bit quantization for memory efficiency
--quant_type nf4                       # NormalFloat4 quantization

--tune_mm_mlp_adapter True             # Fine-tune MM projector
--tune_deepfake_mlp_adapter True       # Fine-tune KAN projector
--freeze_backbone False                # Allow LoRA training

--learning_rate 2e-5                   # Lower LR for fine-tuning
--num_train_epochs 1                   # 1 epoch is usually enough
--per_device_train_batch_size 4        # Smaller batch (LoRA needs more memory)
```

### Expected Training Time
- **~2-3 hours** on RTX 4080 SUPER (16GB)
- VRAM Usage: ~12-14 GB

### Output
- **Checkpoint:** `./checkpoints/llava-v1.5-7b-deepfake-kan-lora-stage3/`
- This is the final model ready for inference

---

## Training Workflow

### Prerequisites
1. Stage 1 binary detector weights at `./utils/weights/M2F2_Det_densenet121.pth`
2. Base LLaVA model initialized at `./checkpoints/llava-v1.5-7b-deepfake-rand-proj-v1`
3. Training data at `./utils/DDVQA_split/c40/train_DDVQA_format.json`

### Step-by-Step

```bash
# Step 1: Stage 2 - Train KAN Projector Only
cd /root/2026_Cincinnati_Project
bash scripts/kan/stage_2_kan_only.sh

# Wait for training to complete (~3-4 hours)
# Check output: ./checkpoints/llava-v1.5-7b-deepfake-kan-stage2/

# Step 2: Stage 3 - LoRA Fine-tuning
bash scripts/kan/stage_3_lora_finetune.sh

# Wait for training to complete (~2-3 hours)
# Check output: ./checkpoints/llava-v1.5-7b-deepfake-kan-lora-stage3/

# Step 3: Inference
bash scripts/kan/inference_det.sh
bash scripts/kan/inference_exp.sh
```

---

## Comparison with Combined Training

### Original Approach (`stage_2_3_combined.sh`)
- Trains KAN projector + LoRA simultaneously in one stage
- Faster overall (single training run)
- May lead to unstable training due to simultaneous optimization

### Two-Stage Approach (Recommended)
- **Stage 2:** Dedicated KAN projector training
- **Stage 3:** LLM fine-tuning with pre-trained KAN
- More stable convergence
- Better separation of concerns
- Easier to debug and tune each component

---

## Parameter Tuning Guide

### If KAN Projector Underfits (Stage 2)
```bash
--learning_rate 1e-3              # Increase LR
--num_train_epochs 5              # More epochs
--kan_hidden_dim 256              # Larger hidden dimension
```

### If LoRA Underfits (Stage 3)
```bash
--lora_r 128                      # Increase rank
--lora_alpha 256                  # Increase alpha
--learning_rate 5e-5              # Higher LR
```

### If Out of Memory
```bash
# Stage 2:
--per_device_train_batch_size 4   # Reduce batch size
--gradient_accumulation_steps 32  # Increase accumulation

# Stage 3:
--bits 4                          # Use 4-bit quantization
--per_device_train_batch_size 2   # Reduce batch size
```

---

## Monitoring Training

### Key Metrics to Watch

**Stage 2 (KAN Training):**
- `train/loss`: Should decrease steadily to ~0.5-0.8
- `train/learning_rate`: Should follow cosine decay

**Stage 3 (LoRA Fine-tuning):**
- `train/loss`: Should decrease to ~0.3-0.5
- Watch for overfitting if loss plateaus too early

### Using TensorBoard
```bash
tensorboard --logdir ./checkpoints/llava-v1.5-7b-deepfake-kan-stage2/
tensorboard --logdir ./checkpoints/llava-v1.5-7b-deepfake-kan-lora-stage3/
```

---

## Troubleshooting

### Issue: "CUDA out of memory" in Stage 2
**Solution:** Reduce batch size or enable gradient checkpointing
```bash
--per_device_train_batch_size 4
--gradient_checkpointing True  # Already enabled
```

### Issue: "CUDA out of memory" in Stage 3
**Solution:** Use 4-bit quantization (already enabled) or reduce LoRA rank
```bash
--bits 4
--lora_r 32  # Reduce from 64
```

### Issue: Stage 3 loss doesn't decrease
**Solution:** Check if Stage 2 checkpoint is properly loaded
```bash
ls -lh ./checkpoints/llava-v1.5-7b-deepfake-kan-stage2/pytorch_model.bin
```

### Issue: KAN projector outputs NaN
**Solution:** Reduce learning rate or check input normalization
```bash
--learning_rate 1e-4  # Lower LR
```

---

## Expected Results

### Detection Performance (on FF++ test set)
- **AUC:** ~95-97%
- **Accuracy:** ~93-95%

### Explanation Performance (on DDVQA)
- **Detection Accuracy:** ~92-94%
- **BLEU-4:** ~15-20%
- **ROUGE-L:** ~35-40%

---

## File Structure

```
scripts/kan/
├── README_TWO_STAGE.md          # This file
├── stage_2_kan_only.sh          # Stage 2: KAN training only
├── stage_3_lora_finetune.sh     # Stage 3: LoRA fine-tuning
├── stage_2_3_combined.sh        # Original combined training (backup)
├── inference_det.sh             # Detection inference
├── inference_exp.sh             # Explanation inference
└── eval.sh                      # Evaluation script
```

---

## Citation

If you use this two-stage training approach, please cite:

```bibtex
@inproceedings{m2f2det2025,
  title={Rethinking Vision-Language Model in Face Forensics: Multi-Modal Interpretable Forged Face Detector},
  author={...},
  booktitle={CVPR},
  year={2025}
}
```
