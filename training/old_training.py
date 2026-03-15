# training/train_ryuugpt_v3.py
"""
Upgraded RyuuGPT trainer (v3) - patched.
- Pad-aware (uses tokenizer pad id = 0)
- Compatible with RyuuGPT.forward(...) -> (logits, loss, value, reasoning)
- Optional reasoning head integration: logs reasoning loss + metrics
- SafeBucketSampler V2 for RAM-friendly bucketing
- EMA, cosine scheduler with warmup, TF32, BF16/FP16, OOM recovery

  """

import os
import sys
import math
import glob
import time
import random
import logging
import argparse
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

# ---------------------------------------------------
# Project root & model imports
# ---------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from model.Ryuu_gpt import RyuuGPT
from model.config import RyuuGPTConfig

# ---------------------------------------------------
# Optional: reasoning head (if you still want external)
# In most cases, reasoning is wired INSIDE RyuuGPT now,
# so the trainer just reads the "reasoning" dict from forward().
# ---------------------------------------------------
try:
    from model.core.reasoning_head import ReasoningHeadV3 as ExternalReasoningHead
    HAVE_EXTERNAL_REASONING = True
except Exception:
    HAVE_EXTERNAL_REASONING = False

# Optional imports (PEFT / FlashAttention / Lion)
HAVE_PEPT = False
HAVE_FLASH = False
HAVE_LION = False
try:
    import importlib.util as _iu

    def _has_spec(name: str) -> bool:
        try:
            return _iu.find_spec(name) is not None
        except Exception:
            return False

    HAVE_PEPT = _has_spec("peft")
    HAVE_FLASH = _has_spec("flash_attn")
    HAVE_LION = _has_spec("lion_pytorch")
except Exception:
    HAVE_PEPT = False
    HAVE_FLASH = False
    HAVE_LION = False

# ---------------------------------------------------
# CLI
# ---------------------------------------------------
parser = argparse.ArgumentParser(description="RyuuGPT upgraded trainer (v3) patched")
parser.add_argument("--data_dir", type=str, default="data/open/Ryuu_Developer_v1/tokenized")
parser.add_argument("--shards_dir", type=str, default="data/shards")
parser.add_argument("--tokenizer", type=str, default="tokenizer/bpe_tokenizer_postproc.json")
parser.add_argument("--save_dir", type=str, default="checkpoints")
parser.add_argument("--log_dir", type=str, default="runs/ryuugpt_v3")
parser.add_argument("--batch_size", type=int, default=2)  # good default for 6GB GPU
parser.add_argument("--grad_accum", type=int, default=8)
parser.add_argument("--max_steps", type=int, default=50000)
parser.add_argument("--eval_interval", type=int, default=1000)
parser.add_argument("--save_interval", type=int, default=5000)
parser.add_argument("--warmup_steps", type=int, default=1000)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--context_size", type=int, default=1024)
parser.add_argument("--n_layer", type=int, default=16)
parser.add_argument("--n_head", type=int, default=12)
parser.add_argument("--n_embd", type=int, default=768)
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--use_bf16", type=int, default=1)
parser.add_argument("--mixed_precision", type=int, default=1)
parser.add_argument("--num_workers", type=int, default=(0 if os.name == "nt" else 2))
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--auto_oom_retry", type=int, default=1)
parser.add_argument("--enable_lora", action="store_true")
parser.add_argument("--enable_flash", action="store_true")
parser.add_argument("--enable_checkpointing", action="store_true")
parser.add_argument("--enable_packing", action="store_true")
parser.add_argument("--enable_bucket", action="store_true")
parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "lion", "lookahead"])
parser.add_argument("--token_weighting", action="store_true")
parser.add_argument("--enable_reasoning_head", action="store_true",
                    help="Use model's reasoning head (already wired in RyuuGPT) and log reasoning metrics.")
# weight for auxiliary reasoning loss
parser.add_argument("--reasoning_loss_weight", type=float, default=0.1)
args = parser.parse_args()

# ---------------------------------------------------
# Logging
# ---------------------------------------------------
os.makedirs(args.save_dir, exist_ok=True)
os.makedirs(args.log_dir, exist_ok=True)
logger = logging.getLogger("ryuugpt_trainer_v3")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s INFO %(message)s",
    datefmt="%H:%M:%S"
)
logger.info("Starting RyuuGPT v3 trainer (patched)")

# ---------------------------------------------------
# Repro / device
# ---------------------------------------------------
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Device: {device} | num_workers={args.num_workers}")

# ---------------------------------------------------
# TF32 for Ampere+
# ---------------------------------------------------
if device == "cuda":
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("TF32 enabled")
    except Exception:
        pass

# ---------------------------------------------------
# Special token IDs (from tokenizer JSON you shared)
# ---------------------------------------------------
PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3
USER_ID = 4
ASSISTANT_ID = 5
ENDOFTURN_ID = 6

# ---------------------------------------------------
# Tokenizer loader (optional)
# ---------------------------------------------------
tokenizer = None
BPETokenizer = None
HAVE_TOKENIZER = False
try:
    import importlib.util
    spec = importlib.util.find_spec("utils.bpe_tokenizer_v2")
    if spec is not None:
        mod = __import__("utils.bpe_tokenizer_v2", fromlist=["BPETokenizer"])
        BPETokenizer = getattr(mod, "BPETokenizer", None)
        HAVE_TOKENIZER = BPETokenizer is not None
except Exception:
    HAVE_TOKENIZER = False

if HAVE_TOKENIZER and os.path.exists(args.tokenizer):
    try:
        tokenizer = BPETokenizer.load(args.tokenizer)
        logger.info(f"Loaded tokenizer vocab={tokenizer.vocab_size}")
    except Exception as e:
        logger.warning(f"Failed to load tokenizer: {e}")
else:
    logger.warning("Tokenizer not available; sampling will return <no-tokenizer>")

# ---------------------------------------------------
# Dataset
# ---------------------------------------------------
class TokenizedDataset(Dataset):
    def __init__(self, data_dir: str, split: str = "train", context_size: int = 1024, packing: bool = False):
        self.data_dir = data_dir
        self.split = split
        self.context_size = context_size
        self.packing = packing

        files = sorted(f for f in os.listdir(data_dir) if f.startswith(split) and f.endswith(".bin"))
        if not files:
            raise FileNotFoundError(f"No {split} .bin shards in {data_dir}")
        self.bin_files = [os.path.join(data_dir, f) for f in files]
        self.idx_files = [f.replace(".bin", ".idx") for f in self.bin_files]

        for idx in self.idx_files:
            if not os.path.exists(idx):
                raise FileNotFoundError(f"Missing index file: {idx}")

        self.tokens_shards = []
        self.indices_shards = []
        self.sample_counts = []
        for b, idx in zip(self.bin_files, self.idx_files):
            tokens = np.fromfile(b, dtype=np.uint32)
            indices = np.fromfile(idx, dtype=np.int64)
            self.tokens_shards.append(tokens)
            self.indices_shards.append(indices)
            self.sample_counts.append(len(indices))

        self.total_samples = sum(self.sample_counts)
        self.cum_counts = np.cumsum([0] + self.sample_counts)
        logger.info(f"Loaded {self.total_samples} {split} samples across {len(self.bin_files)} shards")

    def __len__(self):
        return self.total_samples

    def _locate(self, idx: int) -> Tuple[int, int]:
        shard = int(np.searchsorted(self.cum_counts, idx, side="right") - 1)
        local = idx - self.cum_counts[shard]
        return shard, local

    def __getitem__(self, idx: int):
        shard, local = self._locate(idx)
        indices = self.indices_shards[shard]
        tokens = self.tokens_shards[shard]
        start = 0 if local == 0 else int(indices[local - 1])
        end = int(indices[local])
        arr = tokens[start:end]
        if arr.size > self.context_size:
            arr = arr[:self.context_size]
        if arr.size < 2:
            x = np.array([PAD_ID], dtype=np.int64)
            y = np.array([PAD_ID], dtype=np.int64)
        else:
            x = arr[:-1].astype(np.int64)
            y = arr[1:].astype(np.int64)
        return torch.from_numpy(x).long(), torch.from_numpy(y).long()


def pad_collate(batch):
    xs, ys = zip(*batch)
    x_padded = pad_sequence(xs, batch_first=True, padding_value=PAD_ID)
    y_padded = pad_sequence(ys, batch_first=True, padding_value=PAD_ID)
    return x_padded, y_padded

# ---------------------------------------------------
# Safe Bucket Sampler V2 (RAM-safe, cached)
# ---------------------------------------------------
class SafeBucketSampler(Sampler):
    """
    RAM-safe bucket sampler.
    - Reads sequence lengths from .idx files
    - Caches lengths to disk (dataset_name.lengths.npy)
    - Sorts by length and groups into buckets to reduce padding
    """

    def __init__(self, dataset: TokenizedDataset, batch_size: int = 2, shuffle: bool = True, cache_dir: str = "cache"):
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
            random.shuffle(buckets)
        flat = np.concatenate(buckets)
        for i in range(0, len(flat), self.batch_size):
            yield flat[i:i + self.batch_size]

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)

# ---------------------------------------------------
# Scheduler (Warmup + Cosine with restarts)
# ---------------------------------------------------
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

# ---------------------------------------------------
# EMA
# ---------------------------------------------------
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

# ---------------------------------------------------
# Checkpoints
# ---------------------------------------------------
def save_checkpoint(step, model, optimizer, scheduler_obj, scaler, ema: Optional[EMAModel],
                    best_val, path_dir=args.save_dir, is_best=False):
    os.makedirs(path_dir, exist_ok=True)
    tmp = os.path.join(path_dir, f"ckpt_step{step}.pt.tmp")
    final = os.path.join(path_dir, f"ckpt_step{step}.pt")
    ckpt = {
        "step": step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
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


def load_latest_checkpoint(path_dir=args.save_dir, map_location=None):
    ckpts = glob.glob(os.path.join(path_dir, "ckpt_step*.pt"))
    if not ckpts:
        return None
    ckpt_steps = []
    for ckpt in ckpts:
        try:
            step = int(ckpt.split("step")[-1].split(".")[0])
            ckpt_steps.append((step, ckpt))
        except ValueError:
            continue
    if not ckpt_steps:
        return None
    latest = sorted(ckpt_steps, key=lambda x: x[0])[-1][1]
    data = torch.load(latest, map_location=map_location)
    return latest, data

# ---------------------------------------------------
# Precision selection (bf16 vs fp16)
# ---------------------------------------------------
supports_bf16 = False
if device == "cuda":
    try:
        supports_bf16 = torch.cuda.is_bf16_supported()
        logger.info(f"BF16 supported: {supports_bf16}")
    except Exception as e:
        logger.warning(f"BF16 check failed: {e}")

use_bf16 = bool(args.use_bf16 and supports_bf16)
use_fp16 = bool(args.mixed_precision and not use_bf16 and device == "cuda")

if use_bf16:
    autocast_dtype = torch.bfloat16
    scaler = None
    logger.info("Using BF16 autocast")
elif use_fp16:
    autocast_dtype = torch.float16
    scaler = torch.amp.GradScaler()
    logger.info("Using FP16 + GradScaler")
else:
    autocast_dtype = torch.float32
    scaler = None
    logger.info("Using FP32 (no mixed precision)")


def get_autocast():
    if hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(
            device_type="cuda" if device == "cuda" else "cpu",
            dtype=autocast_dtype if device == "cuda" else torch.float32,
        )
    else:
        from torch.cuda.amp import autocast
        return autocast

# ---------------------------------------------------
# Build model
# ---------------------------------------------------
vocab_size = 50000
if tokenizer is not None:
    try:
        vocab_size = int(getattr(tokenizer, "vocab_size", vocab_size))
    except Exception:
        pass

model_cfg = RyuuGPTConfig(
    vocab_size=vocab_size,
    context_size=args.context_size,
    n_layer=args.n_layer,
    n_head=args.n_head,
    n_embd=args.n_embd,
    dropout=args.dropout,
    pad_token_id=PAD_ID,
    bos_token_id=BOS_ID,
    eos_token_id=EOS_ID,
    use_reasoning_head=args.enable_reasoning_head,
)

model = RyuuGPT(model_cfg)

# optional FlashAttention
if args.enable_flash and HAVE_FLASH and hasattr(model, "enable_flash_attention"):
    try:
        model.enable_flash_attention()
        logger.info("FlashAttention enabled in model")
    except Exception as e:
        logger.warning(f"Could not enable FlashAttention: {e}")

# gradient checkpointing
if args.enable_checkpointing and hasattr(model, "enable_gradient_checkpointing"):
    try:
        model.enable_gradient_checkpointing(True)
        logger.info("Gradient checkpointing enabled")
    except Exception as e:
        logger.warning(f"Failed to enable checkpointing: {e}")

model = model.to(device)
logger.info(f"Model params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

# (Optional) external reasoning head, if you still want it
external_reasoning_head = None
if args.enable_reasoning_head and HAVE_EXTERNAL_REASONING:
    external_reasoning_head = ExternalReasoningHead(args.n_embd).to(device)
    logger.info("External ReasoningHeadV3 attached (not usually needed if model has its own).")

# optimizer
decay, no_decay = [], []
for n, p in model.named_parameters():
    if not p.requires_grad:
        continue
    if p.ndim == 1 or "bias" in n or "norm" in n.lower() or "embed" in n.lower():
        no_decay.append(p)
    else:
        decay.append(p)

optim_groups = [
    {"params": decay, "weight_decay": 0.01},
    {"params": no_decay, "weight_decay": 0.0},
]

if args.optimizer == "lion" and HAVE_LION:
    from lion_pytorch import Lion
    optimizer = Lion(optim_groups, lr=args.lr, weight_decay=0.01)
    logger.info("Using Lion optimizer")
else:
    optimizer = optim.AdamW(optim_groups, lr=args.lr, betas=(0.9, 0.95), eps=1e-8)
    logger.info("Using AdamW optimizer")

scheduler = WarmupCosineWithRestarts(optimizer, warmup_steps=args.warmup_steps, T_0=2000, eta_min=1e-6)
ema = EMAModel(model, decay=0.9999, device=device)

# LoRA / PEFT (optional)
if args.enable_lora and HAVE_PEPT:
    try:
        from peft import get_peft_model, LoraConfig
        lora_cfg = LoraConfig(r=8, lora_alpha=32, target_modules=["attn", "ffn"], lora_dropout=0.05)
        model = get_peft_model(model, lora_cfg)
        logger.info("LoRA wrappers applied")
    except Exception as e:
        logger.warning(f"LoRA enable failed: {e}")

# ---------------------------------------------------
# Data loaders
# ---------------------------------------------------
train_ds = TokenizedDataset(args.data_dir, "train", context_size=args.context_size, packing=args.enable_packing)
val_ds = TokenizedDataset(args.data_dir, "test", context_size=args.context_size, packing=args.enable_packing)

if args.enable_bucket:
    train_sampler = SafeBucketSampler(train_ds, batch_size=args.batch_size, shuffle=True)
    base_train_loader = DataLoader(
        train_ds,
        batch_sampler=train_sampler,
        collate_fn=pad_collate,
        num_workers=args.num_workers,
        pin_memory=True,
    )
else:
    base_train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=pad_collate,
        num_workers=args.num_workers,
        pin_memory=True,
    )

base_val_loader = DataLoader(
    val_ds,
    batch_size=args.batch_size,
    shuffle=False,
    collate_fn=pad_collate,
    num_workers=args.num_workers,
    pin_memory=True,
)

writer = SummaryWriter(log_dir=args.log_dir)

# ---------------------------------------------------
# Token weighting (optional)
# ---------------------------------------------------
token_weights = None
if args.token_weighting and tokenizer is not None:
    token_weights = torch.ones(vocab_size, device=device)
    struct_tokens = ["def", ":", "return", "(", ")", "class"]
    for t in struct_tokens:
        try:
            tid = tokenizer.token_to_id(t)
            if tid is not None:
                token_weights[tid] = 1.2
        except Exception:
            pass

# ---------------------------------------------------
# Evaluation helper (LM + reasoning loss)
# ---------------------------------------------------
@torch.no_grad()
def evaluate_one_epoch(model_obj, loader, max_batches=50):
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
        with get_autocast():
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

# ---------------------------------------------------
# Sampling helper
# ---------------------------------------------------
@torch.no_grad()
def sample_text(prompt: str = "def example_fn(x):", max_new: int = 64, temperature: float = 0.8, top_k: int = 50):
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
        eos_token_id=EOS_ID,
    )
    toks = out[0].tolist()
    txt = tokenizer.decode(toks)
    model.train()
    return txt

# ---------------------------------------------------
# Training loop
# ---------------------------------------------------
def train():
    step = 0
    best_val = float("inf")
    micro_batch = args.batch_size
    grad_accum = max(1, args.grad_accum)

    # Use local variables to avoid UnboundLocalError when reassigning
    local_train_loader = base_train_loader
    local_val_loader = base_val_loader

    ck = load_latest_checkpoint(args.save_dir, map_location=device)
    if ck is not None:
        path, data = ck
        try:
            model.load_state_dict(data["model_state"], strict=False)
            optimizer.load_state_dict(data.get("optimizer_state", {}))
            if scaler is not None and data.get("scaler_state") is not None:
                scaler.load_state_dict(data.get("scaler_state"))
            step = int(data.get("step", 0))
            best_val = float(data.get("best_val", best_val))
            logger.info(f"Loaded checkpoint {path} (step {step})")
        except Exception as e:
            logger.warning(f"Could not fully load checkpoint: {e}")

    model.train()
    running_loss = None
    running_reason_loss = None

    pbar = tqdm(total=args.max_steps, initial=step)
    try:
        while step < args.max_steps:
            for xb, yb in local_train_loader:
                try:
                    xb = xb.to(device, non_blocking=True)
                    yb = yb.to(device, non_blocking=True)

                    with get_autocast():
                        # model forward → logits, lm_loss, value, reasoning
                        logits, lm_loss, value, reasoning = model(xb, targets=yb)

                        if lm_loss is None:
                            # Should not happen when targets are provided, but guard anyway
                            lm_loss = F.cross_entropy(
                                logits.view(-1, logits.size(-1)),
                                yb.view(-1),
                                ignore_index=PAD_ID,
                            )

                        # Override LM loss with token weighting if enabled
                        if token_weights is not None:
                            lm_loss = F.cross_entropy(
                                logits.view(-1, logits.size(-1)),
                                yb.view(-1),
                                weight=token_weights,
                                ignore_index=PAD_ID,
                            )

                        # Reasoning loss from model (if any)
                        reasoning_loss = None
                        if isinstance(reasoning, dict):
                            r_loss = reasoning.get("loss", None)
                            if r_loss is not None:
                                reasoning_loss = r_loss

                        total_loss = lm_loss
                        if reasoning_loss is not None and args.reasoning_loss_weight > 0.0:
                            total_loss = total_loss + args.reasoning_loss_weight * reasoning_loss

                    # Normalize for gradient accumulation
                    loss_value = total_loss / grad_accum

                    if scaler is not None:
                        scaler.scale(loss_value).backward()
                    else:
                        loss_value.backward()

                    if (step + 1) % grad_accum == 0:
                        if scaler is not None:
                            try:
                                scaler.unscale_(optimizer)
                            except Exception:
                                pass
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        if scaler is not None:
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        scheduler.step()
                        ema.update()

                    # Update running metrics (EMA style)
                    lm_loss_cpu = lm_loss.item()
                    running_loss = lm_loss_cpu if running_loss is None else 0.99 * running_loss + 0.01 * lm_loss_cpu

                    if reasoning_loss is not None:
                        r_val = reasoning_loss.item() if torch.is_tensor(reasoning_loss) else float(reasoning_loss)
                        running_reason_loss = (
                            r_val if running_reason_loss is None else 0.99 * running_reason_loss + 0.01 * r_val
                        )

                    step += 1
                    pbar.update(1)

                    # TensorBoard logging
                    if step % 10 == 0:
                        if running_loss is not None:
                            writer.add_scalar("train/lm_loss", running_loss, step)
                        if running_reason_loss is not None:
                            writer.add_scalar("train/reasoning_loss", running_reason_loss, step)

                        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], step)
                        if device == "cuda":
                            mem_gb = torch.cuda.memory_allocated() / (1024 ** 3)
                            writer.add_scalar("gpu/memory_gb", mem_gb, step)

                        postfix = {"lm_loss": f"{running_loss:.4f}" if running_loss is not None else "n/a",
                                   "lr": optimizer.param_groups[0]["lr"]}
                        if running_reason_loss is not None:
                            postfix["reason"] = f"{running_reason_loss:.4f}"
                        pbar.set_postfix(postfix)

                    # Evaluation
                    if step % args.eval_interval == 0:
                        val_loss, val_reason_loss = evaluate_one_epoch(model, local_val_loader, max_batches=100)

                        # clamp to avoid overflow
                        safe_train = min(running_loss if running_loss is not None else 50.0, 50.0)
                        safe_val = min(val_loss, 50.0)
                        train_ppl = math.exp(safe_train)
                        val_ppl = math.exp(safe_val)
                        entropy_bits = val_loss / math.log(2)

                        writer.add_scalar("eval/lm_loss", val_loss, step)
                        writer.add_scalar("train/ppl", train_ppl, step)
                        writer.add_scalar("eval/ppl", val_ppl, step)
                        writer.add_scalar("eval/entropy_bits", entropy_bits, step)

                        if val_reason_loss is not None:
                            writer.add_scalar("eval/reasoning_loss", val_reason_loss, step)

                        logger.info(
                            f"Step {step}/{args.max_steps} | "
                            f"train_ema={running_loss:.4f} | val={val_loss:.4f} | "
                            f"train_ppl={train_ppl:.2f} | val_ppl={val_ppl:.2f}"
                        )
                        logger.info(
                            f"Val entropy: {entropy_bits:.4f} | Val ppl (metric): {val_ppl:.2f}"
                        )
                        if val_reason_loss is not None:
                            logger.info(
                                f"Val reasoning_loss: {val_reason_loss:.4f}"
                            )

                        # Save checkpoint & sample
                        is_best = val_loss < best_val
                        best_val = min(best_val, val_loss)
                        save_checkpoint(step, model, optimizer, scheduler, scaler, ema, best_val, is_best=is_best)

                        try:
                            s = sample_text(prompt="def example_fn(x):", max_new=64)
                            writer.add_text("sample", s, step)
                            logger.info(f"[SAMPLE @ step {step}]\n{s[:400]}")
                        except Exception as e:
                            logger.warning(f"sample failed: {e}")

                    # Periodic save
                    if step % args.save_interval == 0:
                        save_checkpoint(step, model, optimizer, scheduler, scaler, ema, best_val)

                    if step >= args.max_steps:
                        break

                except RuntimeError as e:
                    msg = str(e).lower()
                    if "out of memory" in msg or "cuda out of memory" in msg:
                        logger.warning("CUDA OOM detected during training step. Attempting recovery...")
                        if scaler is not None:
                            try:
                                scaler.unscale_(optimizer)
                            except Exception:
                                pass
                        torch.cuda.empty_cache()
                        if args.auto_oom_retry:
                            if grad_accum > 1:
                                grad_accum = max(1, grad_accum // 2)
                                logger.warning(f"Reducing grad_accum to {grad_accum} and retrying.")
                                continue
                            elif micro_batch > 1:
                                micro_batch = max(1, micro_batch // 2)
                                logger.warning(
                                    f"Reducing micro_batch to {micro_batch} and rebuilding loaders."
                                )
                                # rebuild loaders with smaller batch size
                                if args.enable_bucket:
                                    sampler = SafeBucketSampler(train_ds, batch_size=micro_batch, shuffle=True)
                                    local_train_loader = DataLoader(
                                        train_ds,
                                        batch_sampler=sampler,
                                        collate_fn=pad_collate,
                                        num_workers=args.num_workers,
                                        pin_memory=True,
                                    )
                                else:
                                    local_train_loader = DataLoader(
                                        train_ds,
                                        batch_size=micro_batch,
                                        shuffle=True,
                                        collate_fn=pad_collate,
                                        num_workers=args.num_workers,
                                        pin_memory=True,
                                    )
                                local_val_loader = DataLoader(
                                    val_ds,
                                    batch_size=micro_batch,
                                    shuffle=False,
                                    collate_fn=pad_collate,
                                    num_workers=args.num_workers,
                                    pin_memory=True,
                                )
                                continue
                        logger.error("OOM recovery failed -> re-raising")
                        raise
                    else:
                        raise

            # end epoch
        logger.info("Training finished")
    except KeyboardInterrupt:
        logger.warning("Interrupted by user - saving checkpoint")
        save_checkpoint(step, model, optimizer, scheduler, scaler, ema, best_val)
    finally:
        writer.close()
        pbar.close()

# ---------------------------------------------------
# Entry point
# ---------------------------------------------------
if __name__ == "__main__":
    print("================================================")
    print("    Starting RyuuGPT v3 Reasoning Training      ")
    print("================================================\n")
    train()

