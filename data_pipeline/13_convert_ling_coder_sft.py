"""Convert Ling-Coder-SFT parquet shards into JSONL for pipeline ingestion.

Input:
  data/raw/hf/Ling-Coder-SFT/data/*.parquet
Output:
  data/raw/hf/ling_coder_sft/train.jsonl
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq


def iter_parquet_rows(path: Path):
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=2000):
        yield batch.to_pydict()


def normalize_messages(messages):
    if not isinstance(messages, list):
        return None
    user = None
    assistant = None
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role", "")).lower()
        content = m.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        if role in ("user", "human") and user is None:
            user = content.strip()
        elif role in ("assistant", "gpt", "bot") and user is not None:
            assistant = content.strip()
            break
    if user and assistant:
        return {"prompt": user, "response": assistant}
    return None


def main() -> int:
    in_dir = Path("data/raw/hf/Ling-Coder-SFT/data")
    out_path = Path("data/raw/hf/ling_coder_sft/train.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob("*.parquet"))
    if not files:
        print("[ERR] No parquet files found")
        return 1

    kept = 0
    scanned = 0
    with out_path.open("w", encoding="utf-8") as f:
        for fp in files:
            for batch in iter_parquet_rows(fp):
                msgs = batch.get("messages")
                if msgs is None:
                    continue
                for messages in msgs:
                    scanned += 1
                    rec = normalize_messages(messages)
                    if rec is None:
                        continue
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    kept += 1
                    if kept % 50000 == 0:
                        print(f"[OK] kept={kept} scanned={scanned}")

    print(f"[DONE] Ling-Coder-SFT: kept={kept} scanned={scanned} -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
