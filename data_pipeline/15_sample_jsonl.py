"""Sample a fixed number of rows from a large JSONL file (streaming).

Usage:
  cuda\Scripts\python.exe data_pipeline/15_sample_jsonl.py \
    --input data/raw/hf/ling_coder_sft/train.jsonl \
    --output data/raw/hf/ling_coder_sft/train_1M.jsonl \
    --limit 1000000 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser("Sample JSONL")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Reservoir sampling for uniform sample without loading all data.
    sample = []
    seen = 0
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            seen += 1
            if len(sample) < args.limit:
                sample.append(obj)
            else:
                j = random.randint(0, seen - 1)
                if j < args.limit:
                    sample[j] = obj
            if seen % 500000 == 0:
                print(f"[OK] scanned={seen} sampled={len(sample)}")

    with out_path.open("w", encoding="utf-8") as f:
        for obj in sample:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"[DONE] sampled={len(sample)} scanned={seen} -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
