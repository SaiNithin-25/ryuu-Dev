"""Run the full Ryuu-Dev data pipeline end-to-end with strict failure handling."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


STAGES = [
    "01_collect_sources.py",
    "02_clean_normalize.py",
    "03_quality_filter.py",
    "04_deduplicate.py",
    "05_split_dataset.py",
    "06_train_tokenizer.py",
    "07_tokenize_and_shard.py",
    "08_build_dpo_seed.py",
    "09_validate_pipeline.py",
]


def run_stage(root: Path, script_name: str, python_exe: str) -> float:
    script = root / script_name
    if not script.exists():
        raise FileNotFoundError(f"Missing stage script: {script}")

    project_root = root.parent
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(project_root) + (( ";" + existing) if existing else "")

    print(f"[RUN] {script_name}")
    t0 = time.time()
    result = subprocess.run([python_exe, str(script)], cwd=str(project_root), env=env)
    dt = time.time() - t0
    if result.returncode != 0:
        raise RuntimeError(f"Stage failed: {script_name} (exit={result.returncode})")
    print(f"[OK]  {script_name} in {dt:.1f}s")
    return dt


def read_json(path: Path):
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser("Run full data pipeline")
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to use for stage scripts",
    )
    parser.add_argument(
        "--report_path",
        default="data/open/Ryuu_Developer_v1/pipeline_report.json",
        help="Where to write pipeline report JSON",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    total = 0.0
    stage_times = {}

    try:
        for stage in STAGES:
            dt = run_stage(root, stage, args.python)
            stage_times[stage] = dt
            total += dt
    except Exception as exc:
        print(f"[ERR] Pipeline aborted: {exc}")
        return 1

    print(f"[DONE] Full pipeline completed in {total:.1f}s")
    # Build concise report
    report = {
        "total_seconds": total,
        "stages": stage_times,
        "python": args.python,
        "project_root": str(root.parent),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    stats_path = root.parent / "data" / "open" / "Ryuu_Developer_v1" / "stats.json"
    tok_stats_path = root.parent / "data" / "open" / "Ryuu_Developer_v1" / "tokenized" / "tokenization_stats.json"
    stats = read_json(stats_path)
    tok_stats = read_json(tok_stats_path)
    if stats:
        report["dataset_stats"] = stats
        print(f"[SUMMARY] train={stats.get('num_train')} test={stats.get('num_test')} total={stats.get('num_total')}")
    if tok_stats:
        report["tokenization_stats"] = tok_stats
        train_tokens = tok_stats.get("train", {}).get("tokens")
        test_tokens = tok_stats.get("test", {}).get("tokens")
        if train_tokens is not None and test_tokens is not None:
            print(f"[SUMMARY] tokens(train/test)={train_tokens}/{test_tokens}")

    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[SUMMARY] report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
