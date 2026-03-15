import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

KEY_MODULES = [
    "checks.check_power",
    "checks.check_shards",
    "checks.check_ckpt",
    "checks.simple_check",
    "checks.test_config",
    "testing.test_tok",
    "testing.test_tokenizer_consistency",
    "testing.test_protocol_inference",
    "testing.test2",
    "testing.sanity_check_ryuugpt_v4",
    "testing.test_checkpoint_matrix",
    "testing.test_inference_smoke",
    "testing.test_inference_deterministic",
    "testing.reasoning_test",
    "testing.test_trainer_smoke",
]


def run_module(module: str, python_bin: str, timeout_sec: int) -> tuple[bool, float]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)

    cmd = [python_bin, "-m", module]
    start = time.time()
    try:
        proc = subprocess.run(cmd, cwd=str(ROOT), env=env, timeout=timeout_sec)
        ok = proc.returncode == 0
    except subprocess.TimeoutExpired:
        ok = False
    elapsed = time.time() - start
    return ok, elapsed


def terminate_after_completion(failed: list[str], total: int):
    print("=== Summary ===")
    print(f"Passed: {total - len(failed)}")
    print(f"Failed: {len(failed)}")

    if failed:
        print("Failed modules:")
        for mod in failed:
            print(f"  - {mod}")
        print("Terminating runner with exit code 1.")
        sys.exit(1)

    print("All key checks/tests passed.")
    print("Terminating runner with exit code 0.")
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser("Run all key RyuuAI checks/tests")
    parser.add_argument("--python", default=sys.executable, help="Python interpreter path")
    parser.add_argument("--timeout", type=int, default=300, help="Per-module timeout in seconds")
    parser.add_argument("--skip", nargs="*", default=[], help="Modules to skip")
    args = parser.parse_args()

    selected = [m for m in KEY_MODULES if m not in set(args.skip)]

    print("=== RyuuAI Key Checks/Test Runner ===")
    print(f"Python: {args.python}")
    print(f"Project root: {ROOT}")
    print(f"Total modules: {len(selected)}")
    print()

    failed = []
    for i, mod in enumerate(selected, start=1):
        print(f"[{i}/{len(selected)}] Running {mod} ...")
        ok, elapsed = run_module(mod, args.python, args.timeout)
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {mod} ({elapsed:.1f}s)")
        print()
        if not ok:
            failed.append(mod)

    terminate_after_completion(failed, len(selected))


if __name__ == "__main__":
    main()
