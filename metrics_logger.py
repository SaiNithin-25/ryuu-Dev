from __future__ import annotations

import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EXPECTED_TAGS = (
    "train/loss_ema",
    "train/ppl_ema",
    "train/reason_ema",
    "val/loss",
    "val/ppl",
    "val/entropy",
    "val/reason",
    "perf/tok_per_step",
    "perf/tok_per_sec",
    "perf/tokens_k",
)

EMPTY_SCALARS = pd.DataFrame(columns=["tag", "step", "value", "wall_time", "run_name"])
EMPTY_CHECKPOINTS = pd.DataFrame(
    columns=["name", "kind", "step", "size_mb", "updated_at", "updated_ts", "path"]
)


@dataclass
class RunSnapshot:
    run_name: str
    latest_step: int | None = None
    max_steps: int | None = None
    completion_ratio: float | None = None
    last_update_ts: float | None = None
    freshness: str = "missing"
    train_loss: float | None = None
    val_loss: float | None = None
    best_val_loss: float | None = None
    best_step: int | None = None
    train_ppl: float | None = None
    val_ppl: float | None = None
    train_reason: float | None = None
    val_reason: float | None = None
    tok_per_step: float | None = None
    tok_per_sec: float | None = None
    total_tokens: float | None = None
    generalization_gap: float | None = None
    val_improvement_pct: float | None = None
    val_velocity: float | None = None
    steps_per_second: float | None = None
    eta_seconds: float | None = None
    checkpoint_count: int = 0
    latest_checkpoint_step: int | None = None
    best_checkpoint_present: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _patch_numpy_for_tensorboard() -> None:
    if not hasattr(np, "string_"):
        np.string_ = np.bytes_
    if not hasattr(np, "unicode_"):
        np.unicode_ = np.str_


def discover_run_dirs(log_root: str | Path = "runs") -> list[Path]:
    root = Path(log_root)
    if not root.exists():
        return []

    run_dirs: dict[Path, float] = {}
    for event_file in root.rglob("events.out.tfevents.*"):
        try:
            run_dirs[event_file.parent] = max(
                run_dirs.get(event_file.parent, 0.0),
                event_file.stat().st_mtime,
            )
        except OSError:
            continue

    return [path for path, _ in sorted(run_dirs.items(), key=lambda item: item[1], reverse=True)]


def infer_checkpoint_dir(
    run_dir: str | Path,
    checkpoint_root: str | Path = "checkpoints",
) -> Path | None:
    run_path = Path(run_dir)
    candidate = Path(checkpoint_root) / run_path.name
    if candidate.exists():
        return candidate
    return None


def load_scalar_frame(log_dir: str | Path) -> pd.DataFrame:
    log_path = Path(log_dir)
    if log_path.is_file():
        log_path = log_path.parent
    if not log_path.exists():
        return EMPTY_SCALARS.copy()

    _patch_numpy_for_tensorboard()
    from tensorboard.backend.event_processing import event_accumulator as tb_event_accumulator

    accumulator = tb_event_accumulator.EventAccumulator(
        str(log_path),
        size_guidance={tb_event_accumulator.SCALARS: 0},
    )
    accumulator.Reload()

    frames: list[pd.DataFrame] = []
    for tag in accumulator.Tags().get("scalars", []):
        events = accumulator.Scalars(tag)
        if not events:
            continue
        frames.append(
            pd.DataFrame(
                {
                    "tag": tag,
                    "step": [int(event.step) for event in events],
                    "value": [float(event.value) for event in events],
                    "wall_time": [float(event.wall_time) for event in events],
                    "run_name": log_path.name,
                }
            )
        )

    if not frames:
        return EMPTY_SCALARS.copy()

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["tag", "step", "wall_time"])
    df = df.drop_duplicates(subset=["tag", "step"], keep="last")
    df = df.reset_index(drop=True)
    return df


def pivot_scalar_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["step", "wall_time", *EXPECTED_TAGS])

    deduped = frame.sort_values(["step", "wall_time", "tag"]).drop_duplicates(
        subset=["tag", "step"],
        keep="last",
    )
    values = deduped.pivot(index="step", columns="tag", values="value")
    wall_time = deduped.groupby("step", as_index=True)["wall_time"].max()
    wide = pd.concat([wall_time, values], axis=1).reset_index().sort_values("step")
    wide.columns.name = None
    return wide


def load_checkpoint_index(checkpoint_dir: str | Path | None) -> pd.DataFrame:
    if checkpoint_dir is None:
        return EMPTY_CHECKPOINTS.copy()

    ckpt_path = Path(checkpoint_dir)
    if not ckpt_path.exists():
        return EMPTY_CHECKPOINTS.copy()

    records: list[dict[str, Any]] = []
    for path in sorted(ckpt_path.glob("ckpt_step*.pt")):
        step = _parse_step_from_name(path.name)
        records.append(_checkpoint_record(path, "step", step))

    best_path = ckpt_path / "ckpt_best.pt"
    if best_path.exists():
        records.append(_checkpoint_record(best_path, "best", None))

    if not records:
        return EMPTY_CHECKPOINTS.copy()

    df = pd.DataFrame.from_records(records)
    return df.sort_values(["updated_ts", "step"], ascending=[False, False]).reset_index(drop=True)


def build_run_snapshot(
    frame: pd.DataFrame,
    checkpoint_dir: str | Path | None = None,
    max_steps: int | None = None,
) -> dict[str, Any]:
    run_name = frame["run_name"].iloc[0] if not frame.empty else Path(checkpoint_dir or "run").name
    snapshot = RunSnapshot(run_name=run_name, max_steps=max_steps if max_steps and max_steps > 0 else None)

    if frame.empty:
        checkpoints = load_checkpoint_index(checkpoint_dir)
        snapshot.checkpoint_count = int(len(checkpoints))
        snapshot.latest_checkpoint_step = _safe_int(checkpoints["step"].max()) if not checkpoints.empty else None
        snapshot.best_checkpoint_present = bool((checkpoints["kind"] == "best").any()) if not checkpoints.empty else False
        return snapshot.to_dict()

    wide = pivot_scalar_frame(frame)
    checkpoints = load_checkpoint_index(checkpoint_dir)

    snapshot.latest_step = _safe_int(wide["step"].max())
    snapshot.last_update_ts = _safe_float(wide["wall_time"].max())
    snapshot.freshness = _freshness_label(snapshot.last_update_ts)
    snapshot.checkpoint_count = int(len(checkpoints))
    snapshot.latest_checkpoint_step = _safe_int(checkpoints["step"].max()) if not checkpoints.empty else None
    snapshot.best_checkpoint_present = bool((checkpoints["kind"] == "best").any()) if not checkpoints.empty else False

    if snapshot.max_steps and snapshot.latest_step is not None and snapshot.max_steps > 0:
        snapshot.completion_ratio = min(1.0, max(0.0, snapshot.latest_step / snapshot.max_steps))

    last_row = wide.iloc[-1]
    snapshot.train_loss = _row_value(last_row, "train/loss_ema")
    snapshot.val_loss = _row_value(last_row, "val/loss")
    snapshot.train_ppl = _row_value(last_row, "train/ppl_ema")
    snapshot.val_ppl = _row_value(last_row, "val/ppl")
    snapshot.train_reason = _row_value(last_row, "train/reason_ema")
    snapshot.val_reason = _row_value(last_row, "val/reason")
    snapshot.tok_per_step = _row_value(last_row, "perf/tok_per_step")
    snapshot.tok_per_sec = _row_value(last_row, "perf/tok_per_sec")

    token_series = _series(wide, "perf/tokens_k")
    if not token_series.empty:
        snapshot.total_tokens = float(token_series.sum() * 1000.0)

    if snapshot.train_loss is not None and snapshot.val_loss is not None:
        snapshot.generalization_gap = float(snapshot.train_loss - snapshot.val_loss)

    val_rows = _series_frame(wide, "val/loss")
    if not val_rows.empty:
        best_row = val_rows.loc[val_rows["value"].idxmin()]
        snapshot.best_val_loss = _safe_float(best_row["value"])
        snapshot.best_step = _safe_int(best_row["step"])
        first_val = _safe_float(val_rows.iloc[0]["value"])
        if first_val not in (None, 0.0) and snapshot.val_loss is not None:
            snapshot.val_improvement_pct = float(((first_val - snapshot.val_loss) / first_val) * 100.0)
        if len(val_rows) >= 2:
            snapshot.val_velocity = float(val_rows["value"].iloc[-1] - val_rows["value"].iloc[-2])

    snapshot.steps_per_second = _step_rate(wide)
    if (
        snapshot.max_steps
        and snapshot.latest_step is not None
        and snapshot.steps_per_second
        and snapshot.latest_step < snapshot.max_steps
    ):
        snapshot.eta_seconds = float((snapshot.max_steps - snapshot.latest_step) / snapshot.steps_per_second)

    return snapshot.to_dict()


def summarize_available_metrics(frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return []
    return sorted(frame["tag"].unique().tolist())


def _checkpoint_record(path: Path, kind: str, step: int | None) -> dict[str, Any]:
    stat = path.stat()
    return {
        "name": path.name,
        "kind": kind,
        "step": step,
        "size_mb": round(stat.st_size / (1024 * 1024), 2),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime)),
        "updated_ts": float(stat.st_mtime),
        "path": str(path),
    }


def _parse_step_from_name(name: str) -> int | None:
    stem = Path(name).stem
    if "step" not in stem:
        return None
    raw = stem.split("step", 1)[-1]
    return int(raw) if raw.isdigit() else None


def _series(wide: pd.DataFrame, column: str) -> pd.Series:
    if column not in wide.columns:
        return pd.Series(dtype=float)
    return wide[column].dropna()


def _series_frame(wide: pd.DataFrame, column: str) -> pd.DataFrame:
    if column not in wide.columns:
        return pd.DataFrame(columns=["step", "value"])
    subset = wide[["step", column]].dropna().rename(columns={column: "value"})
    return subset.reset_index(drop=True)


def _row_value(row: pd.Series, column: str) -> float | None:
    value = row.get(column)
    return _safe_float(value)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _freshness_label(last_update_ts: float | None) -> str:
    if last_update_ts is None:
        return "missing"
    age_seconds = max(0.0, time.time() - last_update_ts)
    if age_seconds < 15 * 60:
        return "live"
    if age_seconds < 6 * 60 * 60:
        return "recent"
    return "stale"


def _step_rate(wide: pd.DataFrame) -> float | None:
    if len(wide) < 2:
        return None

    clock = wide[["step", "wall_time"]].dropna().sort_values("step").reset_index(drop=True)
    if len(clock) < 2:
        return None

    delta_steps = clock["step"].diff()
    delta_time = clock["wall_time"].diff()
    rates = delta_steps / delta_time
    rates = rates.replace([np.inf, -np.inf], np.nan).dropna()
    rates = rates[rates > 0]
    if rates.empty:
        return None
    return float(rates.tail(5).median())
