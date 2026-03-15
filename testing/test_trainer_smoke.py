"""
tools/test_trainer_smoke.py

Quick smoke-test for the modular RyuuGPT trainer.

Usage:
    python tools/test_trainer_smoke.py

This will:
 - create a tiny synthetic shard dataset under ./data/test_shards/
 - run the Trainer for a small number of steps (default 20)
 - save checkpoints to ./checkpoints/test_smoke/
 - remove the synthetic data when finished (optional)
"""

import os
import shutil
import numpy as np
import argparse
import subprocess
import time
import tempfile
from pathlib import Path

# Ensure project root is importable
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# constants for fake data
VOCAB_SIZE = 2000
PAD_ID = 0
BOS_ID = 2
EOS_ID = 3


def safe_copy(src, dst, retries=5, delay=0.1):
    for _ in range(retries):
        try:
            shutil.copyfile(src, dst)
            return
        except PermissionError:
            time.sleep(delay)
    shutil.copyfile(src, dst)


def create_shards(out_dir: str, num_shards: int = 2, samples_per_shard: int = 20, max_len: int = 24):
    """
    Create small .bin/.idx shards compatible with TokenizedDataset in training/data.py

    Each shard will contain `samples_per_shard` sequences of random length (2..max_len).
    .bin contains uint32 token ids concatenated; .idx contains int64 cumulative end offsets.
    """
    os.makedirs(out_dir, exist_ok=True)
    for s in range(num_shards):
        tokens_list = []
        idx_list = []
        cur = 0
        for i in range(samples_per_shard):
            L = np.random.randint(2, max_len + 1)  # at least 2 tokens
            # generate tokens in range [4, VOCAB_SIZE-1]
            toks = np.random.randint(4, VOCAB_SIZE, size=(L,), dtype=np.uint32)
            tokens_list.append(toks)
            cur += L
            idx_list.append(cur)

        tokens_arr = np.concatenate(tokens_list).astype(np.uint32)
        idx_arr = np.array(idx_list, dtype=np.int64)

        bin_path = os.path.join(out_dir, f"train_shard{s}.bin")
        idx_path = bin_path.replace(".bin", ".idx")
        tokens_arr.tofile(bin_path)
        idx_arr.tofile(idx_path)
        print(f"Created shard: {bin_path} ({tokens_arr.size} tokens, {idx_arr.size} samples)")


def create_test_dirs(base="data/test_shards"):
    # Remove if exists (be careful!)
    base = os.path.abspath(base)
    if os.path.exists(base):
        try:
            shutil.rmtree(base)
        except PermissionError:
            base = tempfile.mkdtemp(prefix="test_shards_", dir=os.path.dirname(base))
            print(f"Base shard dir was locked; using fallback: {base}")
    os.makedirs(base, exist_ok=True)

    # Create 2 train shards in one call, then convert the 2nd into test shard.
    create_shards(os.path.join(base), num_shards=2, samples_per_shard=12, max_len=16)

    # find produced .bin files (should be train_shard0.bin and train_shard1.bin)
    bins = sorted([f for f in os.listdir(base) if f.endswith(".bin")])
    if len(bins) < 2:
        raise RuntimeError("Expected 2 shards, found: " + ", ".join(bins))

    # Keep first as train_shard0.bin, rename second pair to test_shard0.bin
    src_bin = os.path.join(base, bins[1])
    src_idx = src_bin.replace(".bin", ".idx")
    dst_bin = os.path.join(base, "test_shard0.bin")
    dst_idx = os.path.join(base, "test_shard0.idx")
    safe_copy(src_bin, dst_bin)
    safe_copy(src_idx, dst_idx)
    print(f"Prepared test shard: {dst_bin} (copied from {src_bin})")
    return base



def run_smoke_test(tmp_data_dir="data/test_shards", ckpt_dir="checkpoints/test_smoke", logs_dir="runs/test_smoke"):
    # create fake data
    tmp_data_dir = create_test_dirs(tmp_data_dir)

    train_script = ROOT / "training" / "train_v3.1.py"
    if not train_script.exists():
        print(f"Skipping smoke test because {train_script} was not found.")
        return

    cmd = [
        sys.executable,
        str(train_script),
        "--data_dir", tmp_data_dir,
        "--save_dir", ckpt_dir,
        "--log_dir", logs_dir,
        "--batch_size", "2",
        "--grad_accum", "1",
        "--max_steps", "20",
        "--eval_interval", "5",
        "--warmup_steps", "0",
        "--context_size", "32",
        "--n_layer", "2",
        "--n_head", "2",
        "--n_embd", "128",
        "--dropout", "0.0",
    ]

    print("Starting smoke training via train_v3.1.py (very small)...")
    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        raise SystemExit(result.returncode)
    print("Smoke training finished - check logs and checkpoints.")

    # Optionally, cleanup data and ckpts
    # shutil.rmtree(tmp_data_dir)
    # shutil.rmtree(ckpt_dir)
    # shutil.rmtree(logs_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("smoke test trainer")
    parser.add_argument("--data_dir", type=str, default="data/test_shards")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints/test_smoke")
    parser.add_argument("--logs_dir", type=str, default="runs/test_smoke")
    parser.add_argument("--clean", action="store_true", help="remove generated data & checkpoints after run")
    args = parser.parse_args()

    run_smoke_test(tmp_data_dir=args.data_dir, ckpt_dir=args.ckpt_dir, logs_dir=args.logs_dir)

    if args.clean:
        for p in [args.data_dir, args.ckpt_dir, args.logs_dir]:
            if os.path.exists(p):
                shutil.rmtree(p)
                print("Removed", p)
