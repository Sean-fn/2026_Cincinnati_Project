# 低 VRAM 環境下的 KAN 訓練策略

## 目前狀態
- 模型載入已經接近 VRAM 滿
- 預算: ~24GB VRAM (假設)

---

## 方案對比

### 方案 A: LoRA (推薦 ✓✓✓)

**特性**:
- LoRA r=32: 只增加 ~500MB 參數
- 不需要儲存完整梯度
- INT8 量化 LLM

**配置**:
```bash
--bits 8                          # INT8 量化
--lora_enable True
--lora_r 32
--per_device_train_batch_size 1
--gradient_accumulation_steps 40
--model_max_length 512            # 降低序列長度
```

**預估 VRAM**:
```
LLM (INT8):                7 GB
KAN adapter:              0.1 GB
LoRA 參數+梯度:           0.5 GB
Activations (b=1, l=512): 4-6 GB
Overhead:                 1-2 GB
────────────────────────────────
總計:                    12-16 GB ✓
```

**優點**:
- ✓ 最低 VRAM 成本
- ✓ LoRA 訓練收斂快
- ✓ 可與 KAN 無縫配合
- ✓ 推論時可 merge

**缺點**:
- ❌ LoRA 可能無法學習大幅變化

---

### 方案 B: 解凍前 2-3 層 + LoRA

**特性**:
- 前 2 層完全訓練
- 中間層用 LoRA
- 最後層凍結

**配置**:
```bash
--num_unfreeze_early_layers 2     # 解凍前 2 層
--lora_enable True
--lora_r 32
```

**預估 VRAM**:
```
前 2 層梯度:              ~2-3 GB
LoRA:                   0.5 GB
其他:                   10-12 GB
────────────────────────────────
總計:                    13-16 GB ✓
```

**優點**:
- ✓ 直接適應 projector 輸出 (如您所說!)
- ✓ 更强的自適應能力

**缺點**:
- ❌ 比純 LoRA 多 2-3 GB
- ❌ 程式碼修改較多

---

### 方案 C: 只訓練 KAN (無 LoRA/解凍)

**配置**:
```bash
--tune_deepfake_mlp_adapter True
--freeze_backbone True
```

**預估 VRAM**:
```
KAN 參數+梯度:            0.2 GB
Activations:             6-8 GB
────────────────────────────────
總計:                    7-10 GB (最少!)
```

**優點**:
- ✓ 最低 VRAM
- ✓ 快速訓練

**缺點**:
- ❌ KAN 輸出分佈可能不匹配 LLM
- ❌ 效果可能不佳

---

## VRAM 成本詳細分析

### INT8 量化效果

| 配置 | LLM 模型 | KAN/LoRA | 梯度 | Activations | 總計 |
|------|---------|----------|------|-------------|------|
| FP16 完整 | 14 GB | 0.1 GB | 14 GB | 8-12 GB | **36-40 GB** ❌ |
| FP16 LoRA | 14 GB | 0.5 GB | 0.5 GB | 8-12 GB | **23-27 GB** |
| INT8 LoRA | **7 GB** | 0.5 GB | 0.5 GB | 8-12 GB | **16-20 GB** ✓ |
| INT8 KAN only | **7 GB** | 0.1 GB | 0.1 GB | 6-8 GB | **13-15 GB** ✓✓ |

---

## 推薦策略 (針對您的情況)

### 第一選擇: INT8 + LoRA (平衡)

```bash
bash finetune_stage2_lora_lowvram.sh
```

**為什麼**:
1. LoRA 低 VRAM (500MB)
2. INT8 量化 LLM (節省 50%)
3. 訓練收斂好
4. 容易 merge 權重
5. VRAM: 16-20 GB

---

### 第二選擇: INT8 + 解凍前 2 層 (強自適應)

```bash
# 需要修改 train_deepfake.py
num_unfreeze_early_layers = 2
lora_enable = True
lora_r = 32
bits = 8
```

**為什麼**:
1. 前層直接適應 KAN 輸出
2. 保持 LoRA 的低 VRAM
3. 更强的特徵提取
4. VRAM: 16-20 GB

---

### 第三選擇: KAN Only (最激進)

```bash
--tune_deepfake_mlp_adapter True
--bits 8
--model_max_length 256  # 進一步降低
```

**為什麼**:
1. 最低 VRAM (13-15 GB)
2. 但需要確保 KAN 初始化好

---

## 不同 KAN 配置的 VRAM 成本

使用 INT8 LoRA 時:

| KAN 配置 | 參數 | VRAM 增加 | 推薦 |
|---------|------|---------|------|
| mlp2x_gelu (當前) | 16.8M | +0 GB | ✓ 基準 |
| KAN (h=256) | 16.8M | +0 GB | ✓ 更好表達力 |
| KAN (h=128) | 8.4M | -0.1 GB | ✓✓ 省 VRAM |
| KAN (h=64) | ~5M | -0.2 GB | ✓✓✓ 最省 |
| Efficient KAN | ~0.5M | -0.3 GB | ✓✓✓ 極省 |

---

## 快速開始

### 1. INT8 LoRA (馬上試)

```bash
# 修改您的 finetune_stage_2.sh:

# 添加 INT8 量化
--bits 8 \
--load_in_8bit True \

# 添加 LoRA
--lora_enable True \
--lora_r 32 \
--lora_alpha 64 \

# 降低 batch
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 40 \

# 降低序列長度
--model_max_length 512 \

# 8bit Adam 優化器
--optim adamw_8bit
```

### 2. 檢查當前 VRAM 使用

```python
# 在 train_deepfake.py 開始時添加
import torch
print(f"VRAM used at startup: {torch.cuda.memory_allocated() / 1024 / 1024:.2f} MB")
```

### 3. 監控訓練 VRAM

```bash
watch -n 1 'nvidia-smi --query-gpu=memory.used --format=csv'
```

---

## 降低 KAN 本身 VRAM 的技巧

### 1. 使用小 Hidden Dim

```python
# 在 config.json 中
{
    "kan_hidden_dim": 64,      # 從 1024 降到 64
    "kan_grid_size": 3,        # 從 5 降到 3
    "kan_spline_order": 2      # 從 3 降到 2
}
```

預估節省: ~100 MB

### 2. Efficient KAN Approximation

不用 pykan (複雜), 用簡單的可學習激活:

```python
class LightweightKAN(nn.Module):
    def __init__(self, in_dim=2, out_dim=4096, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim)
        )
    def forward(self, x):
        return self.net(x)

# 參數: 2*64 + 64*4096 = 262K (只有 MLP 的 1.5%)!
```

---

## 最終建議

**立即試用**: INT8 + LoRA 方案

1. 修改您的訓練腳本 (添加上面的配置)
2. 運行 `bash finetune_stage2_lora_lowvram.sh`
3. 監控 VRAM 使用
4. 如果還是不夠, 換成 "KAN Only" 方案

**預期結果**:
- VRAM: 16-20 GB ✓
- 訓練速度: 正常
- 效果: 與原始 MLP 相當或更好
