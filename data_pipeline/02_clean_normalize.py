"""Normalize heterogeneous rows to {prompt, response}."""

import json
from collections import defaultdict
from multiprocessing import Pool, cpu_count

from data_pipeline.common import load_config, read_jsonl, normalize_row


def _normalize_worker(row):
    norm = normalize_row(row)
    if norm is None:
        return None
    src = row.get("_source", "unknown")
    norm["_source"] = src
    norm["_source_file"] = row.get("_source_file", "")
    return norm


def main():
    cfg = load_config()
    in_path = cfg["output"]["collected"]
    out_path = cfg["output"]["cleaned"]

    dropped = 0
    source_kept = defaultdict(int)
    source_dropped = defaultdict(int)
    perf = cfg.get("performance", {})
    workers = max(1, int(perf.get("workers", 1)))
    chunk_size = max(100, int(perf.get("chunk_size", 2000)))

    with open(out_path, "w", encoding="utf-8") as out:
        if workers == 1:
            for row in read_jsonl(in_path):
                norm = normalize_row(row)
                if norm is None:
                    dropped += 1
                    src = row.get("_source", "unknown")
                    source_dropped[src] += 1
                    continue
                src = row.get("_source", "unknown")
                norm["_source"] = src
                norm["_source_file"] = row.get("_source_file", "")
                source_kept[src] += 1
                out.write(json.dumps(norm, ensure_ascii=False) + "\n")
        else:
            pool = Pool(processes=min(workers, cpu_count()))
            try:
                buf = []
                for row in read_jsonl(in_path):
                    buf.append(row)
                    if len(buf) >= chunk_size:
                        results = pool.map(_normalize_worker, buf)
                        for i, norm in enumerate(results):
                            if norm is None:
                                dropped += 1
                                src = buf[i].get("_source", "unknown")
                                source_dropped[src] += 1
                                continue
                            src = norm.get("_source", "unknown")
                            source_kept[src] += 1
                            out.write(json.dumps(norm, ensure_ascii=False) + "\n")
                        buf = []
                if buf:
                    results = pool.map(_normalize_worker, buf)
                    for i, norm in enumerate(results):
                        if norm is None:
                            dropped += 1
                            src = buf[i].get("_source", "unknown")
                            source_dropped[src] += 1
                            continue
                        src = norm.get("_source", "unknown")
                        source_kept[src] += 1
                        out.write(json.dumps(norm, ensure_ascii=False) + "\n")
            finally:
                pool.close()
                pool.join()

    print(f"[OK] Cleaned rows: {sum(source_kept.values())}")
    print(f"[OK] Dropped rows: {dropped}")
    print("[OK] Source kept:")
    for k in sorted(source_kept):
        print(f"  - {k}: {source_kept[k]}")
    if dropped:
        print("[OK] Source dropped:")
        for k in sorted(source_dropped):
            print(f"  - {k}: {source_dropped[k]}")
    print(f"[OK] Output     : {out_path}")


if __name__ == "__main__":
    main()
