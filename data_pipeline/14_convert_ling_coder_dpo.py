"""Convert Ling-Coder-DPO parquet into JSONL for DPO training.

Input:
  data/raw/Ling-Coder-DPO/*.parquet
Output:
  data/dpo/ling_coder_dpo.jsonl
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq


def iter_parquet_rows(path: Path):
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=2000):
        yield batch.to_pydict()


def main() -> int:
    in_dir = Path("data/raw/Ling-Coder-DPO")
    out_path = Path("data/dpo/ling_coder_dpo.jsonl")
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
                prompts = batch.get("prompt")
                chosens = batch.get("chosen")
                rejecteds = batch.get("rejected")
                if prompts is None or chosens is None or rejecteds is None:
                    continue
                n = min(len(prompts), len(chosens), len(rejecteds))
                for i in range(n):
                    scanned += 1
                    p = prompts[i]
                    c = chosens[i]
                    r = rejecteds[i]
                    if not (isinstance(p, str) and isinstance(c, str) and isinstance(r, str)):
                        continue
                    if not (p.strip() and c.strip() and r.strip()):
                        continue
                    rec = {"prompt": p.strip(), "chosen": c.strip(), "rejected": r.strip()}
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    kept += 1
                    if kept % 50000 == 0:
                        print(f"[OK] kept={kept} scanned={scanned}")

    print(f"[DONE] Ling-Coder-DPO: kept={kept} scanned={scanned} -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
