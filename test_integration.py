from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import torch

from metrics_logger import (
    build_run_snapshot,
    infer_checkpoint_dir,
    load_checkpoint_index,
    load_scalar_frame,
    pivot_scalar_frame,
)


def _patch_numpy_aliases() -> None:
    if not hasattr(np, "string_"):
        np.string_ = np.bytes_
    if not hasattr(np, "unicode_"):
        np.unicode_ = np.str_


def _run_integration_check(tmp_path: Path) -> None:
    _patch_numpy_aliases()
    from torch.utils.tensorboard import SummaryWriter

    log_dir = tmp_path / "runs" / "demo_run"
    ckpt_dir = tmp_path / "checkpoints" / "demo_run"
    log_dir.mkdir(parents=True)
    ckpt_dir.mkdir(parents=True)

    writer = SummaryWriter(str(log_dir))
    for step, train_loss, val_loss, tok_per_sec in (
        (250, 4.8, 4.4, 1100.0),
        (500, 4.1, 3.8, 1250.0),
        (750, 3.7, 3.3, 1325.0),
    ):
        writer.add_scalar("train/loss_ema", train_loss, step)
        writer.add_scalar("train/ppl_ema", np.exp(train_loss), step)
        writer.add_scalar("val/loss", val_loss, step)
        writer.add_scalar("val/ppl", np.exp(val_loss), step)
        writer.add_scalar("perf/tok_per_sec", tok_per_sec, step)
        writer.add_scalar("perf/tok_per_step", tok_per_sec / 5.0, step)
        writer.add_scalar("perf/tokens_k", 40.0 + step / 100.0, step)
    writer.close()

    torch.save({"step": 750, "best_val": 3.3}, ckpt_dir / "ckpt_step750.pt")
    torch.save({"step": 750, "best_val": 3.3}, ckpt_dir / "ckpt_best.pt")

    frame = load_scalar_frame(log_dir)
    wide = pivot_scalar_frame(frame)
    checkpoints = load_checkpoint_index(ckpt_dir)
    snapshot = build_run_snapshot(frame, checkpoint_dir=ckpt_dir, max_steps=1000)

    assert not frame.empty
    assert len(wide) == 3
    assert set(wide["step"].tolist()) == {250, 500, 750}
    assert len(checkpoints) == 2
    assert infer_checkpoint_dir(log_dir, tmp_path / "checkpoints") == ckpt_dir

    assert snapshot["latest_step"] == 750
    assert math.isclose(snapshot["best_val_loss"], 3.3, rel_tol=1e-6)
    assert snapshot["best_step"] == 750
    assert snapshot["completion_ratio"] == 0.75
    assert snapshot["latest_checkpoint_step"] == 750
    assert snapshot["best_checkpoint_present"] is True
    assert math.isclose(snapshot["tok_per_sec"], 1325.0, rel_tol=1e-6)


def test_metrics_pipeline_reads_training_logs(tmp_path) -> None:
    _run_integration_check(tmp_path)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp_dir:
        _run_integration_check(Path(tmp_dir))
    print("integration_ok")
