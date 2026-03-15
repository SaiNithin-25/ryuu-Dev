"""Collect raw datasets from configured sources into a unified JSONL."""

import json
import time
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Tuple

from data_pipeline.common import load_config, iter_source_files, iter_rows_from_file, ensure_parent


def _process_file(args: Tuple[str, str, str]) -> Tuple[str, int, str]:
    src_name, file_path, out_dir = args
    out_path = Path(out_dir) / f"{src_name}_{abs(hash(file_path))}.jsonl"
    count = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for row in iter_rows_from_file(file_path, "auto"):
            if not isinstance(row, dict):
                continue
            row["_source"] = src_name
            row["_source_file"] = file_path
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return src_name, count, str(out_path)


def main():
    cfg = load_config()
    out_path = cfg["output"]["collected"]
    ensure_parent(out_path)
    Path(out_path).write_text("", encoding="utf-8")

    total_files = 0
    total_rows = 0
    per_source_rows = defaultdict(int)
    per_source_files = defaultdict(int)

    caps = cfg.get("source_caps", {})
    perf = cfg.get("performance", {})
    workers = max(1, int(perf.get("workers", 1)))
    for src in cfg["sources"]:
        files = iter_source_files(src)
        print(f"[SRC] {src['name']} -> {len(files)} files")
        total_files += len(files)
        per_source_files[src["name"]] += len(files)

        cap = caps.get(src["name"])
        if isinstance(cap, int) and cap <= 0:
            print(f"[SKIP] {src['name']} cap={cap}")
            continue

        # If cap is set, process sequentially to enforce limit precisely.
        if isinstance(cap, int) and cap > 0:
            for f in files:
                batch = []
                for row in iter_rows_from_file(f, src.get("format", "auto")):
                    if not isinstance(row, dict):
                        continue
                    row["_source"] = src["name"]
                    row["_source_file"] = f
                    batch.append(row)
                if batch:
                    with open(out_path, "a", encoding="utf-8") as out:
                        for row in batch:
                            out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    total_rows += len(batch)
                    per_source_rows[src["name"]] += len(batch)
                    if per_source_rows[src["name"]] >= cap:
                        print(f"[CAP] {src['name']} reached cap={cap}")
                        break
            continue

        # Parallel per-file processing for uncapped sources.
        if files:
            temp_dir = Path(out_path).parent / f"collect_parts_{int(time.time())}_{src['name']}"
            temp_dir.mkdir(parents=True, exist_ok=True)
            if workers > 1 and len(files) > 1:
                pool = Pool(processes=min(workers, cpu_count()))
                try:
                    tasks = [(src["name"], f, str(temp_dir)) for f in files]
                    results = pool.map(_process_file, tasks)
                finally:
                    pool.close()
                    pool.join()
            else:
                results = [_process_file((src["name"], f, str(temp_dir))) for f in files]

            with open(out_path, "a", encoding="utf-8") as out:
                for src_name, count, part in results:
                    if count <= 0:
                        continue
                    with open(part, "r", encoding="utf-8") as pf:
                        for line in pf:
                            out.write(line)
                    total_rows += count
                    per_source_rows[src_name] += count

    print(f"[OK] Collected files: {total_files}")
    print(f"[OK] Collected rows : {total_rows}")
    print("[OK] Source breakdown:")
    for name in sorted(per_source_rows):
        print(f"  - {name}: {per_source_rows[name]} rows from {per_source_files[name]} files")
    print(f"[OK] Output        : {out_path}")

    empty_sources = [k for k, v in per_source_files.items() if v == 0]
    if empty_sources:
        print("[WARN] Sources with 0 files:")
        for s in sorted(empty_sources):
            print(f"  - {s}")

    if total_rows == 0:
        print("[ERR] No raw data found. Add dataset files under data/raw/... and rerun.")
        raise SystemExit(1)
    if total_rows < cfg.get("min_total_samples", 1000):
        print(f"[WARN] Collected rows below min_total_samples ({cfg.get('min_total_samples', 1000)})")


if __name__ == "__main__":
    main()
