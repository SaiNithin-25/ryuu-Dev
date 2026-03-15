"""Split deduped dataset into train/test."""

import json
import hashlib
from collections import defaultdict
from data_pipeline.common import load_config, read_jsonl, write_jsonl, set_seed, canonicalize_text


def main():
    cfg = load_config()
    set_seed(cfg["seed"])

    in_path = cfg["output"]["deduped"]
    train_path = cfg["output"]["train"]
    test_path = cfg["output"]["test"]
    stats_path = cfg["output"]["stats"]

    rows = list(read_jsonl(in_path))

    import random
    random.shuffle(rows)

    # Leakage-safe split: all identical prompts map to the same side.
    train = []
    test = []
    train_ratio = cfg["split"]["train_ratio"]
    for row in rows:
        key = canonicalize_text(row["prompt"])
        h = int(hashlib.sha1(key.encode("utf-8")).hexdigest(), 16) % 10_000
        if (h / 10_000.0) < train_ratio:
            train.append(row)
        else:
            test.append(row)

    # Keep at least one test sample if data is small.
    if not test and train:
        test.append(train.pop())

    # Guarantee minimum test rows by moving deterministic tail rows.
    min_test = max(1, int((1.0 - train_ratio) * len(rows)))
    if len(test) < min_test and len(train) > min_test:
        need = min_test - len(test)
        test.extend(train[-need:])
        del train[-need:]

    write_jsonl(train_path, train)
    write_jsonl(test_path, test)

    by_source_train = defaultdict(int)
    by_source_test = defaultdict(int)
    for row in train:
        by_source_train[row.get("_source", "unknown")] += 1
    for row in test:
        by_source_test[row.get("_source", "unknown")] += 1

    stats = {
        "num_total": len(rows),
        "num_train": len(train),
        "num_test": len(test),
        "train_ratio": train_ratio,
        "source_train_counts": dict(by_source_train),
        "source_test_counts": dict(by_source_test),
    }

    from data_pipeline.common import ensure_parent
    ensure_parent(stats_path)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"[OK] Train: {len(train)}")
    print(f"[OK] Test : {len(test)}")
    print("[OK] Train source distribution:")
    for s, c in sorted(by_source_train.items()):
        print(f"  - {s}: {c}")
    print("[OK] Test source distribution:")
    for s, c in sorted(by_source_test.items()):
        print(f"  - {s}: {c}")
    print(f"[OK] Stats: {stats_path}")


if __name__ == "__main__":
    main()
