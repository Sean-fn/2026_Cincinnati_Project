# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

M2F2-Det (Multi-Modal Face Forgery Detector) is a deepfake detection system built on LLaVA-v1.5 that combines CLIP-based vision-language learning with custom deepfake detection models. The project performs binary classification (real/fake) and generates natural language explanations for its decisions.

Paper: "Rethinking Vision-Language Model in Face Forensics: Multi-Modal Interpretable Forged Face Detector" (CVPR 2025 Oral)

## Environment Setup

Create the conda environment:
```bash
conda env create -f environment.yml
conda activate M2F2_det
```

Key dependencies:
- PyTorch 2.0.1 with CUDA 11.7 (`torch==2.0.1+cu117`)
- Torchvision 0.16.2
- LLaVA 1.2.2.post1
- Transformers 4.37.0
- DeepSpeed 0.12.6 (for distributed training)
- Flash Attention 2.5.7

## Architecture Overview

The system uses a **three-stage training pipeline**:

### Stage 1: Binary Detector Training
- Trains the M2F2Det model (`sequence/models/M2F2_Det/models/model.py`) which combines:
  - CLIP text encoder with learnable prompt tokens
  - CLIP vision encoder (optional, can reuse LLaVA's encoder)
  - DenseNet121 or EfficientNet backbone for deepfake detection
  - Multi-modal fusion layers
- Outputs: `.pth` checkpoint files containing the binary detector weights

### Stage 2: Multi-Modal Alignment
- Initializes LLaVADeepfakeCasualLM by merging LLaVA-1.5-7b with Stage-1 weights
- Trains custom MLP adapter layers (`deepfake_mlp_adapter`) to bridge the detector and LLM
- Key config requirement: `mm_vision_select_feature` must be `"cls_patch"` (not `"patch"`)
- Uses DeepSpeed Zero-2 for distributed training

### Stage 3: LoRA Fine-tuning
- Applies LoRA (r=128, alpha=256) to the language model
- Tunes both mm_projector and deepfake_mlp_adapter
- Outputs delta weights that must be merged with Stage-2 weights for inference

## Training Commands

### Stage 1: Train Binary Detector
```bash
bash stage_1_train.sh
```
Runs `stage_1_detection.py` which trains the M2F2Det model on FF++ dataset.

### Stage 2: Multi-Modal Alignment

First, merge LLaVA-1.5-7b with Stage-1 weights:
```bash
python scripts/merge_lora_weights_deepfake_random.py \
  --model-path /path/to/llava-v1.5-7b \
  --save-model-path ./checkpoints/llava-1.5-7b-deepfake-rand-proj-v1
```

Then run training:
```bash
bash stage_2_train.sh
```

This script:
1. Calls `scripts/merge_lora_weights_deepfake_random.py` to initialize weights
2. Runs `scripts/finetune_stage_2.sh` via DeepSpeed
3. Key parameters: `--tune_deepfake_mlp_adapter True`, `--freeze_mm_mlp_adapter True`

### Stage 3: LoRA Fine-tuning

```bash
bash stage_3_train.sh
```

This script:
1. Merges Stage-2 delta weights with base model
2. Runs `scripts/finetune_stage_3.sh` with LoRA enabled
3. Merges final LoRA delta weights to produce M2F2-Det checkpoint

## Inference Commands

### Detection Performance (Binary Classification)
```bash
bash stage_1_inference.sh
```
Runs `stage_1_detection_inference.py` on FF++ test set. Modify line 145 in the script to set the dataset path.

### Explanation Performance (on DDVQA dataset)

First, unzip the DDVQA dataset:
```bash
unzip utils/DDVQA_images/c40.zip -d utils/DDVQA_images/
```

Download M2F2-Det weights from Hugging Face:
```bash
cd checkpoints
git lfs clone https://huggingface.co/CHELSEA234/llava-v1.5-7b-M2F2-Det
cd ..
```

Run detection inference:
```bash
bash stage_3_inference_det.sh
```
Output: `outputs/DDVQA/DDVQA_det_c40.jsonl`

Run explanation inference:
```bash
bash stage_3_inference_exp.sh
```
Output: `outputs/DDVQA/DDVQA_exp_c40.jsonl`

Evaluate results:
```bash
python eval/eval_judgement.py
python eval/eval_explanation.py
```

## Model Architecture Details

### M2F2Det Model (`llava/model/deepfake/M2F2Det/model.py`)

The M2F2Det class combines three modalities:
1. **CLIP Text Encoder**: Uses learnable prompt tokens for "forged" vs "authentic" concepts
2. **CLIP Vision Encoder**: Produces 577 features (1 CLS + 576 patch tokens)
3. **Deepfake Backbone**: DenseNet121 or EfficientNet with custom projection layers

Key components:
- `clip_vision_alpha` and `clip_text_alpha`: Learnable scaling parameters
- `deepfake_proj`: Projects backbone features to hidden_size (1024)
- `output`: Final classifier (2 * 1024 + 576 → 2 classes)

**Critical preprocessing note**: The forward() method expects images preprocessed using LLaVA's CLIP preprocessing pipeline. When integrating into LLaVA, inputs must use identical preprocessing to maintain consistency.

### LLaVA Integration

The system extends LLaVA's architecture (`llava/model/llava_arch.py`) by:
- Adding `deepfake_model_name` and `deepfake_model_path` to config.json
- Changing `mm_vision_select_feature` from `"patch"` to `"cls_patch"` to include CLS token
- Adding custom projector layers that fuse CLIP features with deepfake backbone outputs

## Important Configuration Notes

### Config.json Modifications for Stage 2+

When initializing LLaVADeepfakeCasualLM, modify the base LLaVA-1.5-7b config.json:
```json
{
  "_name_or_path": "LLaVA-1.5-7b",
  "deepfake_model_name": "densenet121",
  "deepfake_model_path": "/path/to/Stage-1-weights.pth",
  "mm_vision_select_feature": "cls_patch"
}
```

### DeepSpeed Configuration

Training scripts use DeepSpeed Zero-2 (`scripts/zero2.json`). Multi-GPU training via:
```bash
deepspeed --include localhost:1,2,3,4,5,7 --master_port 29801 llava/train/train_deepfake.py ...
```

### Training Data Format

Stage 2 & 3 expect JSON files with format:
```json
{
  "image": "relative/path/to/image.jpg",
  "conversations": [
    {"from": "human", "value": "<image>\nQuestion text"},
    {"from": "gpt", "value": "Answer text"}
  ]
}
```

The `image_folder` parameter serves as the base path prefix for image keys.

## Key Files and Directories

- `stage_1_detection.py`: Standalone binary detector training
- `llava/train/train_deepfake.py`: Multi-modal training script for Stages 2 & 3
- `llava/model/deepfake/M2F2Det/model.py`: Core M2F2Det architecture
- `llava/model/llava_arch.py`: LLaVA base architecture (extended for deepfake detection)
- `scripts/merge_lora_weights_deepfake.py`: Merges LoRA/delta weights with base models
- `scripts/merge_lora_weights_deepfake_random.py`: Randomly initializes MLP layers for Stage 2
- `dataset/`: HDF5 dataset loaders for FF++ data
- `utils/weights/`: Pre-trained CLIP encoder (`vision_tower.pth`) and detector weights

## Dataset Notes

- **FF++**: Preprocessed HDF5 files (~300GB for c23+c40), download from Google Drive
- **FF++ test only**: Smaller version (~6.7GB) for quick evaluation
- **DDVQA**: c40 version provided in `utils/DDVQA_images/c40.zip`

The HDF5 format follows naming: `FF++_{manipulation_type}_{compression_rate}.h5`

## Weight Merging Workflow

The three-stage training produces intermediate checkpoints that must be merged:

1. **Stage-1-weights**: Binary detector `.pth` file
2. **Stage-2-init-weights**: LLaVA + Stage-1 + random MLP initialization
3. **Stage-2-weights-Delta**: Trained MLP adapters (delta format)
4. **Stage-2-weights**: Merged Stage-2-init + Stage-2-delta
5. **Stage-3-weights-Delta**: LoRA fine-tuned weights (delta format)
6. **M2F2-Det**: Final merged model for inference

Use `scripts/merge_lora_weights_deepfake.py` for merging operations.

## Gradio Demo

Launch local demo (requires M2F2-Det checkpoint):
```bash
# Terminal 1: Controller
python -m llava.serve.controller --host 0.0.0.0 --port 10000

# Terminal 2: Gradio web server
python -m llava.serve.gradio_web_server --controller http://localhost:10000 --model-list-mode reload

# Terminal 3: Model worker
python -m llava.serve.model_worker --host 0.0.0.0 --controller http://localhost:10000 --port 40000 --worker http://localhost:40000 --model-path /path/to/M2F2-Det

# Access: http://localhost:7860
```

Test with cropped face images from `utils/DDVQA_images/` using prompt: "Determine the authenticity of this image"
