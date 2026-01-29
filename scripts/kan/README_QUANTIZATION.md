# Quantization Guide for KAN Training

## Understanding Precision Levels

### 16-bit (Half Precision) - **NOT Quantization**
- **Data type**: `torch.float16` (fp16) or `torch.bfloat16` (bf16)
- **Native support**: Built into PyTorch/CUDA
- **Memory**: ~50% of fp32
- **Accuracy**: Very close to fp32
- **Requirements**: No special libraries needed
- **Use case**: Standard training on GPUs with enough VRAM

### 8-bit/4-bit Quantization - **True Quantization**
- **Data type**: INT8/INT4 + scaling factors
- **Library**: Requires `bitsandbytes` (and `libcusparse.so.12`)
- **Memory**: 25% (8-bit) or 12.5% (4-bit) of fp32
- **Accuracy**: Slight degradation (NF4 minimizes this)
- **Requirements**: bitsandbytes, proper LD_LIBRARY_PATH
- **Use case**: Low VRAM GPUs (e.g., 16GB RTX 4080 SUPER)

---

## Configuration Comparison

### Current (4-bit QLoRA) - For 16GB GPU
```bash
--bits 4                    # 4-bit quantization
--double_quant True         # Double quantization
--quant_type nf4            # NormalFloat 4-bit
--bf16 True                 # Compute in bf16
--per_device_train_batch_size 6
--gradient_accumulation_steps 20
```
**VRAM**: ~10-12 GB

### 16-bit Training - For High VRAM GPU (64GB+)
```bash
# Remove these 3 lines:
# --bits 4
# --double_quant True
# --quant_type nf4

--bf16 True                 # Keep bf16 for compute
--per_device_train_batch_size 2   # Reduce batch size
--gradient_accumulation_steps 60  # Increase to maintain effective batch
```
**VRAM**: ~40-50 GB

### 8-bit Training - Middle Ground (32GB GPU)
```bash
--bits 8                    # 8-bit instead of 4-bit
--double_quant True         # Keep
--quant_type int8           # INT8 quantization
--bf16 True
--per_device_train_batch_size 4
--gradient_accumulation_steps 30
```
**VRAM**: ~20-25 GB

---

## How to Switch

### To 16-bit (Full Precision):

Edit `scripts/kan/stage_2_3_combined.sh`:

```bash
# Comment out or remove these 3 lines:
# --bits 4 \
# --double_quant True \
# --quant_type nf4 \

# Adjust batch size for higher VRAM usage:
--per_device_train_batch_size 2 \     # Was 6
--gradient_accumulation_steps 60 \    # Was 20 (keep effective batch = 120)
```

### To 8-bit:

```bash
--bits 8 \                  # Change from 4 to 8
--double_quant True \       # Keep
--quant_type int8 \         # Change from nf4 to int8
--per_device_train_batch_size 4 \
--gradient_accumulation_steps 30 \
```

---

## Why bitsandbytes is Only Needed for 4-bit/8-bit

| Feature | 16-bit (fp16/bf16) | 4-bit/8-bit Quantization |
|---------|-------------------|-------------------------|
| Data representation | IEEE 754 half-precision float | Integer + scaling factors |
| PyTorch support | ✓ Native | ✗ Needs bitsandbytes |
| CUDA kernels | ✓ Built-in | ✗ Needs custom kernels |
| libcusparse.so.12 | Not required | **Required** |
| LD_LIBRARY_PATH setup | Not needed | **Needed** |

**Bottom line**:
- 16-bit = Native PyTorch feature (no special setup)
- 4-bit/8-bit = Advanced technique requiring bitsandbytes library

---

## Recommended Configuration by GPU

| GPU VRAM | Recommended | Config |
|----------|-------------|--------|
| 16 GB | 4-bit QLoRA | Current default |
| 24 GB | 8-bit QLoRA | Change to `--bits 8` |
| 40 GB+ | 16-bit | Remove quantization params |
| 80 GB+ | 16-bit, bs=4 | Remove quant, increase batch |

**Your GPU**: NVIDIA H200 NVL (139.8 GB)
- **Recommendation**: Use 16-bit for best accuracy
- You have plenty of VRAM for full precision training

---

## Performance Trade-offs

| Metric | 16-bit | 8-bit | 4-bit |
|--------|--------|-------|-------|
| Accuracy | ✓✓✓ Best | ✓✓ Good | ✓ Acceptable |
| Speed | ✓✓✓ Fastest | ✓✓ Slower | ✓ Slowest |
| VRAM | ✗✗✗ Highest | ✗✗ Medium | ✓✓✓ Lowest |
| Setup complexity | ✓✓✓ Simple | ✗ Complex | ✗✗ Very complex |

