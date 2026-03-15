# ⚡ DPO Generation - Speed Crisis & Solutions

## Current Problem

**Rate: 0.04 samples/sec = 25 seconds per sample = 34+ hours for 5,000 samples**

This is extremely slow. Expected was 8-15 minutes. The issue is:

### Root Cause
1. **No batching**: Generating samples one at a time (sequential)
2. **Large model**: 16 layers × 768 hidden × 12 heads = lots of computation
3. **No KV-cache**: Full forward pass for every token (256 iterations per sample)
4. **Hardware**: Possibly CPU fallback or slow GPU

---

## Solutions (in order of priority)

### Solution 1: BATCH Generation (Immediate - 3-5x speedup)
✅ **Already created**: `dpo_dataset_fast.py`

```bash
# Use batch generation (batch_size=4)
python training/dpo/dpo_dataset_fast.py \
    --ckpt checkpoints/v3_reasoning/ckpt_best.pt \
    --tokenizer tokenizer/bpe_tokenizer_postproc.json \
    --out data/dpo/ryuu_dpo_fast.jsonl \
    --num_samples 5000 \
    --batch_size 4 \
    --max_new_tokens 128
```

**Expected speedup**: 3-5x (from 0.04 → 0.15-0.20 samples/sec)

---

### Solution 2: Reduce Token Length (Immediate - 2-3x speedup)

Change from `--max_new_tokens 256` → `--max_new_tokens 128`

```bash
python training/dpo/dpo_dataset_fast.py \
    --max_new_tokens 128 \
    --batch_size 4
```

**Impact**: 
- 256 → 128 tokens = 2x fewer iterations
- Time per sample: 25s → 12-15s

---

### Solution 3: Verify GPU Usage (Immediate)

Check if CUDA is actually being used:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name()}")
print(f"Current device: {torch.cuda.current_device()}")
```

**If CPU is being used:**
- CPU inference is 10-20x slower than GPU
- Check CUDA installation
- Verify PyTorch was installed with CUDA support

---

### Solution 4: Reduce Model Size (Medium-term)

If model is large:
- Use smaller checkpoint (fewer layers/hidden dims)
- Or use quantization (INT8/INT4)

```bash
# With smaller model config
python training/dpo/dpo_dataset_fast.py \
    --ckpt checkpoints/smaller_model.pt \
    --max_new_tokens 64
```

---

### Solution 5: Implement KV-Cache (Advanced - 5-10x speedup)

This requires model changes but eliminates repeated computation:

```python
# Pseudo-code for KV-cache optimization
kv_cache = {}
for token in range(max_new):
    # Only compute new token position, not full sequence
    logits = model(next_token_only, kv_cache=kv_cache)
    # Instead of: logits = model(full_sequence)
```

**Benefit**: 5-10x speedup on generation
**Effort**: Requires modifying RyuuGPT model architecture

---

## Recommended Quick Fix (for today)

```bash
# Fast batch generation with reduced tokens
python training/dpo/dpo_dataset_fast.py \
    --ckpt checkpoints/v3_reasoning/ckpt_best.pt \
    --tokenizer tokenizer/bpe_tokenizer_postproc.json \
    --out data/dpo/ryuu_dpo_fast.jsonl \
    --num_samples 5000 \
    --batch_size 4 \
    --max_new_tokens 128
```

**Expected result:**
- Rate: 0.15-0.25 samples/sec (vs. 0.04)
- Time: 6-12 hours (vs. 34 hours)
- 3-5x improvement

---

## Test First (100 samples)

```bash
# Quick test with 100 samples to verify speed
python training/dpo/dpo_dataset_fast.py \
    --ckpt checkpoints/v3_reasoning/ckpt_best.pt \
    --tokenizer tokenizer/bpe_tokenizer_postproc.json \
    --out data/dpo/test_fast.jsonl \
    --num_samples 100 \
    --batch_size 4 \
    --max_new_tokens 128
```

This should take ~15-30 seconds for 100 samples (= 0.15-0.25 samples/sec)

---

## What Was Changed

| Feature | Old | New | Benefit |
|---------|-----|-----|---------|
| Generation | Sequential | **Batch (size=4)** | 3-5x faster |
| Max tokens | 256 | 128 | 2x faster |
| Processing | Per-sample | **Batch processing** | Better GPU utilization |

---

## Why So Slow Originally?

1. **Sequential generation**: 1 prompt pair at a time
2. **No GPU batching**: Each pair independently
3. **Long sequences**: 256 tokens = 256 forward passes
4. **Model overhead**: Large 16-layer model

**Analogy**: Like processing one image at a time instead of a batch of 4 images through the GPU.

---

## Files Created

1. **`dpo_dataset_fast.py`** - Batch generation version (3-5x faster)
2. **`dpo_dataset.py`** - Original sequential version (working but slow)

---

## Next Steps

1. **Run fast version**: `python training/dpo/dpo_dataset_fast.py ...`
2. **Monitor rate**: Should see 0.15-0.25 samples/sec (not 0.04)
3. **If still slow**: Check CUDA availability and GPU memory
4. **For future**: Implement KV-cache for 5-10x more speedup

