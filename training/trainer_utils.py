# Utilities extracted from old_training.py to augment train_ryuugpt_v3.py
import os
import glob
import math
import logging
from typing import Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger("trainer_utils")

# -------------------------------
# SafeBucketSampler (RAM-safe)
# -------------------------------
from torch.utils.data import Sampler

class SafeBucketSampler(Sampler):
    def __init__(self, dataset, batch_size: int = 2, shuffle: bool = True, cache_dir: str = "cache"):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        os.makedirs(cache_dir, exist_ok=True)
        key = dataset.data_dir.replace("/", "_").replace("\\", "_")
        self.cache_file = os.path.join(cache_dir, f"{key}.lengths.npy")

        if os.path.exists(self.cache_file):
            logger.info(f"[SafeBucketSampler] Loading cached lengths: {self.cache_file}")
            self.lengths = np.load(self.cache_file)
        else:
            logger.info("[SafeBucketSampler] Building length cache (first run)...")
            self.lengths = self._build_lengths()
            np.save(self.cache_file, self.lengths)
            logger.info(f"[SafeBucketSampler] Cached lengths → {self.cache_file}")

        self.indices = np.arange(len(self.lengths))

    def _build_lengths(self):
        all_lengths = []
        for shard_idx, idx_file in enumerate(self.dataset.idx_files):
            idx_arr = np.fromfile(idx_file, dtype=np.int64)
            shard_lengths = np.diff(idx_arr, prepend=0)
            all_lengths.extend(shard_lengths)
            logger.info(f"  Loaded shard {shard_idx + 1}/{len(self.dataset.idx_files)}: {len(shard_lengths)} lengths")

        all_lengths = np.array(all_lengths, dtype=np.int32)
        logger.info(f"[SafeBucketSampler] Total sequences: {len(all_lengths)}")
        return all_lengths

    def __iter__(self):
        sorted_idx = np.argsort(self.lengths)
        bucket_size = self.batch_size * 50
        buckets = [
            sorted_idx[i:i + bucket_size]
            for i in range(0, len(sorted_idx), bucket_size)
        ]
        if self.shuffle:
            import random
            random.shuffle(buckets)
        flat = np.concatenate(buckets)
        for i in range(0, len(flat), self.batch_size):
            yield flat[i:i + self.batch_size]

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)

# -----------------------------------
# WarmupCosineWithRestarts scheduler
# -----------------------------------
class WarmupCosineWithRestarts:
    def __init__(self, optimizer, warmup_steps, T_0=2000, T_mult=1.5, eta_min=1e-6, max_restarts=10, base_lrs=None):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.max_restarts = max_restarts
        self.T_i = T_0
        self.T_cur = 0
        self.restart_count = 0
        self.base_lrs = base_lrs if base_lrs is not None else [g["lr"] for g in optimizer.param_groups]

    def get_lr(self, progress, base_lr):
        if progress < 0:
            return base_lr * (-progress)
        cos_progress = math.cos(math.pi * progress)
        lr_range = base_lr - self.eta_min
        return self.eta_min + 0.5 * lr_range * (1 + cos_progress)

    def step(self):
        if self.T_cur == 0:
            self.restart_count += 1
        if self.restart_count >= self.max_restarts:
            self.T_i = max(self.T_i, int(sum(self.base_lrs)))
        if self.T_cur < self.warmup_steps:
            progress = -1.0 + float(self.T_cur) / max(1, self.warmup_steps)
        else:
            progress = float(self.T_cur - self.warmup_steps) / float(max(1, self.T_i))
            progress = min(1.0, max(0.0, progress))
        for i, g in enumerate(self.optimizer.param_groups):
            g["lr"] = self.get_lr(progress, self.base_lrs[i])
        self.T_cur += 1
        if self.T_cur >= self.T_i + self.warmup_steps:
            self.T_cur = 0
            self.T_i = int(min(self.T_i * self.T_mult, self.T_i + 10000))

# -------------------------------
# EMAModel
# -------------------------------
class EMAModel:
    def __init__(self, model: torch.nn.Module, decay=0.9999, device: Optional[str] = None):
        self.model = model
        self.decay = decay
        self.device = device or (next(model.parameters()).device)
        self.shadow = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = p.data.clone().detach().to(self.device)

    def update(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = (1.0 - self.decay) * p.data + self.decay * self.shadow[n]

    def apply_shadow(self):
        self._backup = {}
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self._backup[n] = p.data.clone()
                p.data.copy_(self.shadow[n].to(p.device))

    def restore(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                p.data.copy_(self._backup[n].to(p.device))
        self._backup = {}

# -------------------------------
# Autocast helper
# -------------------------------
def get_autocast(autocast_dtype: torch.dtype, device: str):
    if hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(
            device_type="cuda" if device == "cuda" else "cpu",
            dtype=autocast_dtype if device == "cuda" else torch.float32,
        )
    else:
        # Fallback (older torch)
        from torch.cuda.amp import autocast
        return autocast()

# -------------------------------
# Checkpoint helpers
# -------------------------------

def save_checkpoint_enhanced(step: int,
                             model: torch.nn.Module,
                             optimizer: torch.optim.Optimizer,
                             scheduler_obj,
                             scaler: Optional[torch.cuda.amp.GradScaler],
                             ema: Optional[EMAModel],
                             best_val: float,
                             path_dir: str,
                             is_best: bool = False):
    os.makedirs(path_dir, exist_ok=True)
    tmp = os.path.join(path_dir, f"ckpt_step{step}.pt.tmp")
    final = os.path.join(path_dir, f"ckpt_step{step}.pt")
    ckpt = {
        "step": step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": getattr(scheduler_obj, '__dict__', None),
        "best_val": best_val,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
    }
    if ema is not None:
        ckpt["ema_shadow"] = ema.shadow
    torch.save(ckpt, tmp)
    os.replace(tmp, final)
    logger.info(f"Saved checkpoint {final}")
    if is_best:
        best_path = os.path.join(path_dir, "ckpt_best.pt")
        if ema is not None:
            ema.apply_shadow()
            torch.save(model.state_dict(), best_path)
            ema.restore()
        else:
            torch.save(model.state_dict(), best_path)
        logger.info(f"Saved best model to {best_path}")


def load_latest_checkpoint_enhanced(path_dir: str, map_location=None) -> Optional[Tuple[str, dict]]:
    ckpts = glob.glob(os.path.join(path_dir, "ckpt_step*.pt"))
    if not ckpts:
        return None
    ckpt_steps = []
    for ckpt in ckpts:
        try:
            step = int(ckpt.split("step")[-1].split(".")[0])
            ckpt_steps.append((step, ckpt))
        except Exception:
            continue
    if not ckpt_steps:
        return None
    latest = sorted(ckpt_steps, key=lambda x: x[0])[-1][1]
    data = torch.load(latest, map_location=map_location)
    return latest, data

# -------------------------------
# Evaluation helper
# -------------------------------
@torch.no_grad()
def evaluate_one_epoch(model_obj: torch.nn.Module, loader, device: str, max_batches: int = 50, autocast_dtype: Optional[torch.dtype] = None):
    model_obj.eval()
    total_loss = 0.0
    total_reasoning_loss = 0.0
    n = 0
    n_reason = 0

    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x = x.to(device)
        y = y.to(device)
        with get_autocast(autocast_dtype or torch.float32, device):
            logits, loss_lm, value, reasoning = model_obj(x, targets=y)

        if loss_lm is None:
            continue

        total_loss += loss_lm.item()
        n += 1

        if isinstance(reasoning, dict):
            r_loss = reasoning.get("loss", None)
            if r_loss is not None:
                if torch.is_tensor(r_loss):
                    r_loss = r_loss.item()
                total_reasoning_loss += float(r_loss)
                n_reason += 1

    model_obj.train()
    avg_loss = total_loss / max(1, n)
    avg_reason = (total_reasoning_loss / max(1, n_reason)) if n_reason > 0 else None
    return avg_loss, avg_reason

# -------------------------------
# Sampling helper
# -------------------------------
@torch.no_grad()
def sample_text(tokenizer, model, prompt: str = "def example_fn(x):", max_new: int = 64, temperature: float = 0.8, top_k: int = 50, device: str = "cpu", eos_token_id: Optional[int] = None):
    if tokenizer is None:
        return "<no-tokenizer>"
    model.eval()
    ids = tokenizer.encode_ids(prompt)
    ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    out = model.generate(
        ids,
        max_new_tokens=max_new,
        temperature=temperature,
        top_k=top_k,
        do_sample=True,
        eos_token_id=eos_token_id,
    )
    toks = out[0].tolist()
    txt = tokenizer.decode(toks)
    model.train()
    return txt
