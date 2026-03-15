"""Convert local TACO Arrow/Parquet shards into JSONL for pipeline ingestion.

Input:
  data/raw/hf/taco/train/*.arrow
  data/raw/hf/taco/test/*.parquet
Output:
  data/raw/hf/taco/train.jsonl
  data/raw/hf/taco/test.jsonl
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq


def iter_arrow_tables(path: Path):
    with path.open("rb") as f:
        try:
            reader = ipc.RecordBatchFileReader(f)
            for i in range(reader.num_record_batches):
                yield reader.get_batch(i).to_pydict()
            return
        except Exception:
            f.seek(0)
        # Fallback for stream format
        reader = ipc.open_stream(f)
        for batch in reader:
            yield batch.to_pydict()


def iter_parquet_tables(path: Path):
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=5000):
        yield batch.to_pydict()


def parse_json_field(value):
    if isinstance(value, str) and value.strip():
        try:
            return json.loads(value)
        except Exception:
            return value
    return value


def normalize_row(row):
    prompt = row.get("question") or row.get("prompt")
    solutions = row.get("solutions")

    solutions = parse_json_field(solutions)
    if isinstance(solutions, list) and solutions:
        response = solutions[0]
    elif isinstance(solutions, str):
        response = solutions
    else:
        response = row.get("answer")

    if isinstance(prompt, str) and isinstance(response, str) and prompt.strip() and response.strip():
        return {"prompt": prompt.strip(), "response": response.strip()}
    return None


def write_jsonl(rows, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def convert_split(split_dir: Path, out_path: Path):
    rows = []
    files = sorted(split_dir.glob("*.arrow")) + sorted(split_dir.glob("*.parquet"))
    if not files:
        return 0

    for fp in files:
        if fp.suffix == ".arrow":
            tables = iter_arrow_tables(fp)
        else:
            tables = iter_parquet_tables(fp)
        for tbl in tables:
            # tbl is dict of column -> list
            keys = list(tbl.keys())
            n = len(tbl[keys[0]]) if keys else 0
            for i in range(n):
                row = {k: tbl[k][i] for k in keys}
                rec = normalize_row(row)
                if rec is not None:
                    rows.append(rec)

    write_jsonl(rows, out_path)
    return len(rows)


def main() -> int:
    # Support both layouts: train/test or training/tests
    train_dir = Path("data/raw/hf/taco/train")
    test_dir = Path("data/raw/hf/taco/test")
    if not train_dir.exists():
        train_dir = Path("data/raw/hf/taco/training")
    if not test_dir.exists():
        test_dir = Path("data/raw/hf/taco/tests")
    out_train = Path("data/raw/hf/taco/train.jsonl")
    out_test = Path("data/raw/hf/taco/test.jsonl")

    train_count = convert_split(train_dir, out_train)
    test_count = convert_split(test_dir, out_test)

    print(f"[OK] TACO train rows: {train_count} -> {out_train}")
    print(f"[OK] TACO test  rows: {test_count} -> {out_test}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
