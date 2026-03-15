"""Tokenize train/test JSONL and write binary shards (.bin/.idx) for train_v3.1.py."""

import os
import json
import numpy as np

from data_pipeline.common import load_config
from utils.bpe_tokenizer_v2 import BPETokenizer


def iter_records(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            p = obj.get("prompt", "")
            r = obj.get("response", "")
            if p and r:
                yield p, r


def write_split_shards(split_name, in_path, out_dir, tokenizer, tokens_per_shard, context_size):
    os.makedirs(out_dir, exist_ok=True)

    shard_idx = 0
    buf_tokens = []
    sample_ends = []
    cur = 0
    total_samples = 0
    dropped_too_short = 0
    truncated = 0
    total_tokens = 0

    def flush():
        nonlocal shard_idx, buf_tokens, sample_ends, cur
        if not buf_tokens:
            return
        arr = np.array(buf_tokens, dtype=np.uint32)
        idx = np.array(sample_ends, dtype=np.int64)
        bpath = os.path.join(out_dir, f"{split_name}_shard{shard_idx}.bin")
        ipath = os.path.join(out_dir, f"{split_name}_shard{shard_idx}.idx")
        arr.tofile(bpath)
        idx.tofile(ipath)
        print(f"[OK] {bpath} ({len(arr)} tokens, {len(idx)} samples)")
        shard_idx += 1
        buf_tokens = []
        sample_ends = []
        cur = 0

    for p, r in iter_records(in_path):
        text = f"<|user|>\n{p}\n<|endofturn|>\n<|assistant|>\n{r}\n<|endofturn|>"
        ids = tokenizer.encode_ids(text)
        if len(ids) > context_size:
            ids = ids[:context_size]
            truncated += 1
        if len(ids) < 2:
            dropped_too_short += 1
            continue

        if len(buf_tokens) + len(ids) > tokens_per_shard:
            flush()

        buf_tokens.extend(ids)
        cur += len(ids)
        sample_ends.append(cur)
        total_samples += 1
        total_tokens += len(ids)

    flush()
    return {
        "split": split_name,
        "samples": total_samples,
        "tokens": total_tokens,
        "shards": shard_idx,
        "dropped_too_short": dropped_too_short,
        "truncated_to_context": truncated,
    }


def main():
    cfg = load_config()

    tokenizer = BPETokenizer.load(cfg["output"]["tokenizer"])
    out_dir = cfg["output"]["tokenized_dir"]
    tps = cfg["sharding"]["tokens_per_shard"]
    ctx = cfg["sharding"]["context_size"]

    train_stats = write_split_shards("train", cfg["output"]["train"], out_dir, tokenizer, tps, ctx)
    test_stats = write_split_shards("test", cfg["output"]["test"], out_dir, tokenizer, tps, ctx)

    stats_path = os.path.join(out_dir, "tokenization_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump({"train": train_stats, "test": test_stats}, f, indent=2)

    print(f"[OK] Tokenized shards written to: {out_dir}")
    print(f"[OK] Tokenization stats written: {stats_path}")


if __name__ == "__main__":
    main()
