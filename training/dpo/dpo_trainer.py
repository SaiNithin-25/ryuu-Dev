# ============================================================
#                RyuuGPT v3 - DPO TRAINER
# ============================================================

import os
import sys
import json
import argparse
import logging
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

# -------------------------------------------------------------
# FIX PROJECT ROOT PATH
# -------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from model.Ryuu_gpt import RyuuGPT
from model.config import RyuuGPTConfig
from utils.bpe_tokenizer_v2 import BPETokenizer


# -------------------------------------------------------------
# CLI
# -------------------------------------------------------------
parser = argparse.ArgumentParser("RyuuGPT DPO Trainer")
parser.add_argument("--data", type=str, required=True)
parser.add_argument("--ckpt", type=str, required=True)
parser.add_argument("--save_dir", type=str, default="checkpoints/v3_dpo")
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--beta", type=float, default=0.1)
parser.add_argument("--max_steps", type=int, default=3000)
parser.add_argument("--context_size", type=int, default=768)
parser.add_argument("--save_interval", type=int, default=500)
parser.add_argument("--use_bf16", action="store_true")
args = parser.parse_args()

os.makedirs(args.save_dir, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s INFO %(message)s")
logger = logging.getLogger("ryuugpt_dpo")

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------------------------------------
# PRECISION
# -------------------------------------------------------------
use_bf16 = args.use_bf16 and device == "cuda" and torch.cuda.is_bf16_supported()
use_fp16 = device == "cuda" and not use_bf16
scaler = torch.amp.GradScaler("cuda") if use_fp16 else None


def autocast_ctx():
    if device != "cuda":
        return nullcontext()
    return torch.autocast("cuda", torch.bfloat16 if use_bf16 else torch.float16)


# -------------------------------------------------------------
# TOKENIZER
# -------------------------------------------------------------
tokenizer = BPETokenizer.load("tokenizer/bpe_tokenizer_postproc.json")
PAD_ID = 0
EOS_ID = (
    tokenizer.token_to_id("</s>")
    or tokenizer.token_to_id("<eos>")
    or 3
)


def encode_text(prompt, answer):
    ids = tokenizer.encode_ids(prompt + answer)
    ids = ids[:args.context_size]
    return torch.tensor(ids, dtype=torch.long)


def collate(batch):
    chosen_list = []
    rejected_list = []

    for p, c, r in batch:
        chosen_list.append(encode_text(p, c))
        rejected_list.append(encode_text(p, r))

    return (
        pad_sequence(chosen_list, batch_first=True, padding_value=PAD_ID),
        pad_sequence(rejected_list, batch_first=True, padding_value=PAD_ID),
    )


# -------------------------------------------------------------
# DATASET
# -------------------------------------------------------------
class DPODataset(Dataset):
    def __init__(self, path):
        self.samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if all(k in sample for k in ("prompt", "chosen", "rejected")):
                    self.samples.append(sample)

        if not self.samples:
            raise ValueError(f"No valid DPO samples found in {path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        return s["prompt"], s["chosen"], s["rejected"]


# -------------------------------------------------------------
# LOAD MODEL
# -------------------------------------------------------------
logger.info("Loading model checkpoint...")
ckpt = torch.load(args.ckpt, map_location=device)


def _extract_state_dict(data):
    if isinstance(data, dict):
        for key in ("model", "model_state", "state_dict"):
            if key in data and isinstance(data[key], dict):
                return data[key]
    return data


def _infer_cfg_from_state(state):
    emb = state.get("token_emb.weight", None)
    if emb is not None and hasattr(emb, "shape") and len(emb.shape) == 2:
        vocab_size, n_embd = int(emb.shape[0]), int(emb.shape[1])
    else:
        vocab_size, n_embd = tokenizer.vocab_size, 768

    block_keys = [k for k in state.keys() if k.startswith("blocks.")]
    block_ids = [int(k.split(".")[1]) for k in block_keys if len(k.split(".")) > 1 and k.split(".")[1].isdigit()]
    n_layer = max(block_ids) + 1 if block_ids else 16

    mask_key = next((k for k in state.keys() if k.endswith("attn.mask")), None)
    context_size = int(state[mask_key].shape[-1]) if mask_key is not None else args.context_size

    rotary_key = next((k for k in state.keys() if k.endswith("attn.rotary.cos")), None)
    head_dim = int(state[rotary_key].shape[-1]) if rotary_key is not None else 64
    n_head = max(1, n_embd // max(1, head_dim))

    return RyuuGPTConfig(
        vocab_size=vocab_size,
        context_size=context_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        pad_token_id=PAD_ID,
        bos_token_id=2,
        eos_token_id=EOS_ID,
        use_reasoning_head=any(k.startswith("reasoning_head.") for k in state.keys()),
        use_value_head=any(k.startswith("value_head.") for k in state.keys()),
    )


def _align_cfg_with_state(cfg, state):
    emb = state.get("token_emb.weight")
    if emb is not None and hasattr(emb, "shape") and len(emb.shape) == 2:
        cfg.vocab_size = int(emb.shape[0])
        cfg.n_embd = int(emb.shape[1])

    block_ids = []
    for key in state.keys():
        if key.startswith("blocks."):
            parts = key.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                block_ids.append(int(parts[1]))
    if block_ids:
        cfg.n_layer = max(block_ids) + 1

    mask_key = next((k for k in state.keys() if k.endswith("attn.mask")), None)
    if mask_key is not None:
        cfg.context_size = int(state[mask_key].shape[-1])

    rotary_key = next((k for k in state.keys() if k.endswith("attn.rotary.cos")), None)
    if rotary_key is not None:
        head_dim = int(state[rotary_key].shape[-1])
        cfg.n_head = max(1, cfg.n_embd // max(1, head_dim))

    return cfg


def _filter_state_by_shape(model, state):
    model_state = model.state_dict()
    return {
        k: v for k, v in state.items()
        if k in model_state and getattr(v, "shape", None) == model_state[k].shape
    }


state = _extract_state_dict(ckpt)
cfg_dict = (ckpt.get("config") or ckpt.get("model_config")) if isinstance(ckpt, dict) else None
cfg = RyuuGPTConfig(**cfg_dict) if cfg_dict else _infer_cfg_from_state(state)
cfg = _align_cfg_with_state(cfg, state)

model = RyuuGPT(cfg).to(device)
filtered_state = _filter_state_by_shape(model, state)
missing, unexpected = model.load_state_dict(filtered_state, strict=False)

logger.info(f"Loaded checkpoint keys: {len(filtered_state)}/{len(state)}")
logger.info(f"Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")

model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)


# Optional EMA
class EMA:
    def __init__(self, model, decay=0.9999):
        self.shadow = {n: p.detach().clone() for n, p in model.named_parameters()}
        self.decay = decay

    @torch.no_grad()
    def update(self, model):
        for n, p in model.named_parameters():
            self.shadow[n].mul_(self.decay)
            self.shadow[n].add_(p.detach(), alpha=1 - self.decay)


ema = EMA(model)


# -------------------------------------------------------------
# LOG-PROB FUNCTION
# -------------------------------------------------------------
def logprob(model, input_ids):
    logits, _, _, _ = model(input_ids)
    logp = F.log_softmax(logits, dim=-1)

    targets = input_ids[:, 1:]
    logp = logp[:, :-1, :]
    token_logp = torch.gather(logp, -1, targets.unsqueeze(-1)).squeeze(-1)

    mask = (targets != PAD_ID).float()
    return (token_logp * mask).sum(dim=1)


# -------------------------------------------------------------
# TRAINING LOOP
# -------------------------------------------------------------
ds = DPODataset(args.data)
dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)

logger.info(f"Starting DPO Training | {len(ds)} samples")

step = 0
best_loss = float("inf")
pbar = tqdm(total=args.max_steps)

while step < args.max_steps:
    for chosen_ids, rejected_ids in dl:
        chosen_ids = chosen_ids.to(device)
        rejected_ids = rejected_ids.to(device)

        with autocast_ctx():
            lp_c = logprob(model, chosen_ids)
            lp_r = logprob(model, rejected_ids)
            diff = lp_c - lp_r
            loss = -F.logsigmoid(args.beta * diff).mean()

        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        optimizer.zero_grad(set_to_none=True)
        ema.update(model)

        step += 1
        pbar.update(1)
        pbar.set_postfix(loss=f"{loss.item():.4f}")

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_path = os.path.join(args.save_dir, "ckpt_best_dpo.pt")
            torch.save({"model": model.state_dict()}, best_path)
            logger.info(f"Best updated -> {best_path}")

        if step % args.save_interval == 0:
            path = os.path.join(args.save_dir, f"ckpt_dpo_step{step}.pt")
            torch.save({"model": model.state_dict()}, path)
            logger.info(f"Saved {path}")

        if step >= args.max_steps:
            break

logger.info("DPO Training Complete!")
final_path = os.path.join(args.save_dir, "ckpt_final_dpo.pt")
torch.save({"model": model.state_dict()}, final_path)
logger.info(f"Saved final checkpoint -> {final_path}")
pbar.close()
