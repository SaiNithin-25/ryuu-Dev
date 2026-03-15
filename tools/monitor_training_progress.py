import glob
import os
import time
from pathlib import Path

import torch


def latest_step_checkpoint(ckpt_dir: str):
    files = sorted(glob.glob(os.path.join(ckpt_dir, "ckpt_step*.pt")))
    if not files:
        return None
    return max(files, key=lambda p: int(os.path.basename(p).split("step")[-1].split(".")[0]))


def read_ckpt_summary(path: str):
    data = torch.load(path, map_location="cpu")
    step = data.get("step") if isinstance(data, dict) else None
    best_val = data.get("best_val") if isinstance(data, dict) else None
    return step, best_val


def report_once(ckpt_dir: str):
    best_path = os.path.join(ckpt_dir, "ckpt_best.pt")
    latest_step = latest_step_checkpoint(ckpt_dir)

    print(f"Checkpoint dir: {ckpt_dir}")
    if latest_step:
        step, best_val = read_ckpt_summary(latest_step)
        print(f"Latest step checkpoint: {latest_step}")
        print(f"  step={step}, best_val={best_val}")
    else:
        print("Latest step checkpoint: not found yet")

    if os.path.exists(best_path):
        b_step, b_val = read_ckpt_summary(best_path)
        print(f"Best checkpoint: {best_path}")
        print(f"  step={b_step}, best_val={b_val}")
    else:
        print("Best checkpoint: not found yet")


def main():
    import argparse

    parser = argparse.ArgumentParser("Monitor training progress from checkpoints")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/v3_reasoning")
    parser.add_argument("--interval", type=int, default=30, help="Polling interval in seconds")
    parser.add_argument("--loops", type=int, default=1, help="Number of loops; 0 means infinite")
    args = parser.parse_args()

    ckpt_dir = str(Path(args.checkpoint_dir))

    i = 0
    while True:
        print("=" * 60)
        print(time.strftime("%Y-%m-%d %H:%M:%S"))
        report_once(ckpt_dir)
        print("=" * 60)

        i += 1
        if args.loops > 0 and i >= args.loops:
            break
        time.sleep(max(1, args.interval))


if __name__ == "__main__":
    main()
