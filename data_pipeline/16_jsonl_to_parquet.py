"""Convert JSONL to Parquet (optional speed-up for large sources).

Usage:
  cuda\Scripts\python.exe data_pipeline/16_jsonl_to_parquet.py ^
    --input data/raw/custom/math_ai_dataset_2M.jsonl ^
    --output data/raw/custom/math_ai_dataset_2M.parquet
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def main() -> int:
    parser = argparse.ArgumentParser("JSONL -> Parquet")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch_size", type=int, default=10000)
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    writer = None
    total = 0
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if len(rows) >= args.batch_size:
                table = pa.Table.from_pylist(rows)
                if writer is None:
                    writer = pq.ParquetWriter(str(out_path), table.schema)
                writer.write_table(table)
                total += len(rows)
                rows = []

        if rows:
            table = pa.Table.from_pylist(rows)
            if writer is None:
                writer = pq.ParquetWriter(str(out_path), table.schema)
            writer.write_table(table)
            total += len(rows)

    if writer is not None:
        writer.close()
    print(f"[DONE] wrote={total} -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
