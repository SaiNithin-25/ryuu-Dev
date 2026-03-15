# DPO Data Generation Performance Analysis

## Current Bottlenecks Identified

### 1. **🔴 CRITICAL: Double Generation Loop in `dpo_dataset.py`**
**Location:** [training/dpo/dpo_dataset.py](training/dpo/dpo_dataset.py#L158)

**Problem:**
```python
for i in range(args.num_samples):  # e.g., 5000 samples
    prompt = random.choice(PROMPTS)
    a = generate(prompt)  # Full forward pass + token generation
    b = generate(prompt)  # Full forward pass + token generation
    # ...score both...
```

**Impact:** Generating 5,000 samples requires **10,000 generation calls** (each prompt generated twice). Each generation call does:
- Full encoding
- Full model forward pass 
- Token-by-token decoding with `torch.cat()` operations
- Full decoding

**Estimated Time:** At ~100ms per generation on GPU = **1,000 seconds (~16.7 minutes)** for just 5,000 samples.

---

### 2. **🟡 MAJOR: Inefficient Token Generation Loop**
**Location:** [training/dpo/dpo_dataset.py](training/dpo/dpo_dataset.py#L90-L106)

**Problem:**
```python
for _ in range(max_new):
    with torch.no_grad():
        logits, _, _, _ = model(x)  # Full forward pass each iteration!
    next_id = int(torch.argmax(logits[:, -1, :], dim=-1))
    x = torch.cat([x, torch.tensor([[next_id]], device=device)], dim=1)  # Tensor allocation + cat
```

**Issues:**
- Full forward pass for every single token (256 iterations per sample)
- `torch.cat()` creates new tensor each iteration (memory fragmentation)
- Converting logits to scalar then back to tensor (unnecessary conversions)
- No KV-cache optimization for transformer

**Time Cost:** ~50-100ms per 256-token generation

---

### 3. **🟡 Inefficient Scoring Function**
**Location:** [training/dpo/dpo_dataset.py](training/dpo/dpo_dataset.py#L124-L144)

**Problem:**
- Uses hardcoded PROMPTS list with only 7 prompts (too repetitive)
- Scoring is purely rule-based and doesn't correlate with actual quality
- 50% of samples skipped as "ties" (many identical scores)

**Impact:** High number of failed samples = need to generate more pairs

---

### 4. **🟡 Loading Full Dataset into Memory**
**Location:** [training/dpo/train_dpo.py](training/dpo/train_dpo.py#L51-L56)

**Problem:**
```python
class DPODataset(Dataset):
    def __init__(self, path):
        self.samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                self.samples.append(json.loads(line))  # Load ALL samples into RAM
```

**Impact:** For large JSONL files, memory usage can become significant. No lazy loading.

---

### 5. **🟡 Inefficient Collate Function with Padding**
**Location:** [training/dpo/train_dpo.py](training/dpo/train_dpo.py#L80-L88)

**Problem:**
```python
def collate(batch):
    pc, pr = [], []
    for p, c, r in batch:
        pc.append(encode_pair(p, c))  # Re-encode on every batch!
        pr.append(encode_pair(p, r))
    return (
        pad_sequence(pc, batch_first=True, padding_value=PAD_ID),
        pad_sequence(pr, batch_first=True, padding_value=PAD_ID),
    )
```

**Issues:**
- Encoding happens during collation (training step, not preprocessing)
- This means **encoding happens ~60,000 times if training on 5,000 samples with batch_size=2**
- Should be pre-tokenized

---

## Performance Estimates

| Operation | Current | Samples | Total Time |
|-----------|---------|---------|-----------|
| Generate 2 responses per prompt | 100ms × 2 | 5,000 | ~1,000 sec (16.7 min) |
| Encoding during training | 10ms per sample | 5,000 × batches | ~50 sec per epoch |
| Full pipeline | - | 5,000 | ~20+ minutes |

---

## Recommended Optimizations (in priority order)

### Priority 1: Fix Generation Loop
1. **Implement KV-Cache** for transformer to avoid recomputing attention
2. **Batch multiple prompts** instead of serial generation
3. **Use in-place tensor operations** instead of `torch.cat()`
4. **Vectorize token sampling** instead of scalar operations

### Priority 2: Pre-tokenize Dataset
1. **Pre-tokenize DPO dataset** during generation (not during training)
2. **Store as binary shards** like in `07_tokenizer_and_shard.py`
3. **Use memory-mapped files** for large datasets

### Priority 3: Improve Scoring
1. **Use model's confidence scores** instead of rule-based heuristics
2. **Increase prompt diversity** (more than 7 prompts)
3. **Use actual model logits** to determine quality

### Priority 4: Batch Generation
1. **Generate multiple samples per batch** to amortize model loading
2. **Vectorize generation across batch dimension**

### Priority 5: Lazy Loading
1. **Implement lazy loading** in DPODataset
2. **Use IterableDataset** for streaming

---

## Example Optimization for Generation

**Current (~100ms per generation):**
```python
for _ in range(max_new):
    logits, _, _, _ = model(x)  # Full forward pass!
    next_id = int(torch.argmax(logits[:, -1, :], dim=-1))
    x = torch.cat([x, torch.tensor([[next_id]], device=device)], dim=1)
```

**Optimized (~20ms per generation with KV-cache):**
```python
kv_cache = {}
for _ in range(max_new):
    logits = model.generate_with_cache(x[:, -1:], kv_cache)  # Only compute last token!
    next_id = torch.argmax(logits[:, -1, :], dim=-1)
    x = torch.cat([x, next_id.unsqueeze(-1)], dim=1)
```

This alone could reduce generation time **5-10x**.

