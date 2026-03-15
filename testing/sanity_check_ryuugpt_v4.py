# ============================================================
# RyuuGPT v4 — SANITY CHECK
# ============================================================

import torch
import logging
import sys
import os

# ------------------------------------------------------------
# Path setup
# ------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from model.Ryuu_gpt import RyuuGPT
from model.config import RyuuGPTConfig

# ------------------------------------------------------------
# Logging
# ------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [SANITY] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("sanity")

# ------------------------------------------------------------
# Device
# ------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
log.info(f"Using device: {device}")

# ------------------------------------------------------------
# Build small test model
# ------------------------------------------------------------
cfg = RyuuGPTConfig(
    vocab_size=30000,
    context_size=64,
    n_layer=4,
    n_head=4,
    n_embd=256,
    dropout=0.1,
    pad_token_id=0,
    bos_token_id=2,
    eos_token_id=3,
    use_reasoning_head=True,
)

model = RyuuGPT(cfg).to(device)
model.train()

log.info(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

# ------------------------------------------------------------
# Dummy batch
# ------------------------------------------------------------
B, T = 2, 32
x = torch.randint(10, 1000, (B, T), device=device)
y = torch.randint(10, 1000, (B, T), device=device)

# Log average and max sequence lengths (ignoring padding)
avg_len = x.ne(cfg.pad_token_id).sum(dim=1).float().mean()
max_len = x.size(1)
log.info(f"avg_len={avg_len:.1f} max_len={max_len}")

# ------------------------------------------------------------
# Step-aware hooks
# ------------------------------------------------------------
model._current_step = 0
if model.reasoning_head is not None:
    model.reasoning_head._current_step = 0

# ------------------------------------------------------------
# Forward pass
# ------------------------------------------------------------
log.info("Running forward pass...")
logits, lm_loss, _, reasoning = model(x, targets=y)

assert logits.shape == (B, T, cfg.vocab_size)
assert lm_loss is not None
assert torch.isfinite(lm_loss)

log.info(f"LM loss: {lm_loss.item():.4f}")

# ------------------------------------------------------------
# Reasoning head checks
# ------------------------------------------------------------
if reasoning is not None:
    log.info("Reasoning head detected")

    r_loss = reasoning.get("loss")
    r_scores = reasoning.get("scores")

    log.info(f"Reasoning loss: {r_loss}")
    log.info(f"Reasoning scores shape: {tuple(r_scores.shape)}")

    if r_loss is not None:
        assert torch.isfinite(r_loss)

# ------------------------------------------------------------
# Backward pass
# ------------------------------------------------------------
log.info("Running backward pass...")
total_loss = lm_loss
if reasoning and reasoning.get("loss") is not None:
    total_loss = total_loss + reasoning["loss"]

total_loss.backward()

log.info("Backward pass OK")

# ------------------------------------------------------------
# Generation test
# ------------------------------------------------------------
model.eval()
with torch.no_grad():
    prompt = torch.tensor([[2, 100, 200, 300]], device=device)
    out = model.generate(prompt, max_new_tokens=16)



log.info(f"Generated tokens: {out.tolist()}")

log.info("======================================")
log.info("SANITY CHECK PASSED ✔")
log.info("======================================")
