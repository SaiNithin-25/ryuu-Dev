"""Validate the rebuilt dataset artifacts before training."""

import os
import json
import hashlib
from data_pipeline.common import load_config, read_jsonl


def count_lines(path):
    if not os.path.exists(path):
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def check_jsonl_schema(path, keys):
    bad = 0
    total = 0
    for row in read_jsonl(path):
        total += 1
        if not all(k in row and isinstance(row[k], str) and row[k].strip() for k in keys):
            bad += 1
    return total, bad


def check_shards(tokenized_dir):
    bins = [x for x in os.listdir(tokenized_dir) if x.endswith('.bin')] if os.path.exists(tokenized_dir) else []
    idxs = [x for x in os.listdir(tokenized_dir) if x.endswith('.idx')] if os.path.exists(tokenized_dir) else []
    return len(bins), len(idxs)


def prompt_overlap_ratio(train_path, test_path):
    train_prompts = set()
    test_prompts = set()
    for row in read_jsonl(train_path):
        p = row.get("prompt", "").strip().lower()
        if p:
            train_prompts.add(hashlib.sha1(p.encode("utf-8")).hexdigest())
    for row in read_jsonl(test_path):
        p = row.get("prompt", "").strip().lower()
        if p:
            test_prompts.add(hashlib.sha1(p.encode("utf-8")).hexdigest())
    if not test_prompts:
        return 0.0
    overlap = len(train_prompts.intersection(test_prompts))
    return overlap / max(1, len(test_prompts))


def source_coverage(path):
    covered = set()
    for row in read_jsonl(path):
        src = row.get("_source", "").strip()
        if src:
            covered.add(src)
    return covered


def check_dpo_schema(path):
    if not os.path.exists(path):
        return 0, 0
    keys = ["prompt", "chosen", "rejected"]
    total = 0
    bad = 0
    for row in read_jsonl(path):
        total += 1
        if not all(k in row and isinstance(row[k], str) and row[k].strip() for k in keys):
            bad += 1
    return total, bad


def main():
    cfg = load_config()
    val_cfg = cfg.get("validation", {})

    train_path = cfg["output"]["train"]
    test_path = cfg["output"]["test"]
    tok_path = cfg["output"]["tokenizer"]
    tok_dir = cfg["output"]["tokenized_dir"]
    dpo_seed = cfg["output"]["dpo_seed"]

    print("[CHECK] Dataset splits")
    tr_total, tr_bad = check_jsonl_schema(train_path, ["prompt", "response"])
    te_total, te_bad = check_jsonl_schema(test_path, ["prompt", "response"])
    print(f"  train: {tr_total} rows, bad={tr_bad}")
    print(f"  test : {te_total} rows, bad={te_bad}")

    print("[CHECK] Prompt overlap")
    overlap = prompt_overlap_ratio(train_path, test_path)
    print(f"  overlap_ratio(test in train): {overlap:.6f}")

    if val_cfg.get("require_source_coverage", True):
        print("[CHECK] Source coverage")
        train_sources = source_coverage(train_path)
        test_sources = source_coverage(test_path)
        print(f"  train sources: {sorted(train_sources)}")
        print(f"  test sources : {sorted(test_sources)}")
    else:
        train_sources, test_sources = set(), set()

    print("[CHECK] Tokenizer")
    exists = os.path.exists(tok_path)
    print(f"  exists: {exists} -> {tok_path}")
    if exists:
        print(f"  mtime : {os.path.getmtime(tok_path)}")

    print("[CHECK] Shards")
    b, i = check_shards(tok_dir)
    print(f"  .bin: {b}, .idx: {i} in {tok_dir}")

    print("[CHECK] DPO seed")
    dpo_total, dpo_bad = check_dpo_schema(dpo_seed)
    print(f"  dpo pairs: {dpo_total}, bad={dpo_bad} -> {dpo_seed}")

    ok = (
        (tr_total + te_total) >= int(cfg.get("min_total_samples", 1000))
        and
        tr_total >= int(val_cfg.get("min_train_rows", 100))
        and te_total >= int(val_cfg.get("min_test_rows", 10))
        and tr_bad == 0 and te_bad == 0
        and os.path.exists(tok_path) and b > 0 and i > 0
        and overlap <= float(val_cfg.get("max_prompt_overlap_ratio", 0.01))
        and dpo_bad == 0
    )
    if val_cfg.get("require_source_coverage", True):
        ok = ok and (len(train_sources) > 0) and (len(test_sources) > 0)

    if ok:
        print("[OK] Pipeline validation passed")
    else:
        print("[ERR] Pipeline validation failed")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
