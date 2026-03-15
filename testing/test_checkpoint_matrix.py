import glob
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.Ryuu_gpt import RyuuGPT
from model.config import RyuuGPTConfig
from utils.bpe_tokenizer_v2 import BPETokenizer


def candidate_checkpoints():
    candidates = [
        ROOT / "checkpoints" / "ckpt_best.pt",
        ROOT / "checkpoints" / "test_smoke" / "ckpt_best.pt",
        ROOT / "checkpoints" / "phase3_train_smoke" / "ckpt_best.pt",
    ]

    step_ckpts = sorted(glob.glob(str(ROOT / "checkpoints" / "**" / "ckpt_step*.pt"), recursive=True))
    if step_ckpts:
        candidates.append(Path(max(step_ckpts, key=lambda p: int(os.path.basename(p).split("step")[-1].split(".")[0]))))

    uniq = []
    seen = set()
    for p in candidates:
        s = str(p)
        if p.exists() and s not in seen:
            uniq.append(p)
            seen.add(s)
    return uniq


def extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for key in ("model", "model_state", "state_dict"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
    return ckpt


def infer_cfg_from_state(state, default_vocab_size: int) -> RyuuGPTConfig:
    emb = state.get("token_emb.weight")
    if emb is not None and hasattr(emb, "shape") and len(emb.shape) == 2:
        vocab_size, n_embd = int(emb.shape[0]), int(emb.shape[1])
    else:
        vocab_size, n_embd = default_vocab_size, 768

    block_ids = []
    for k in state.keys():
        if k.startswith("blocks."):
            parts = k.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                block_ids.append(int(parts[1]))
    n_layer = max(block_ids) + 1 if block_ids else 16

    rotary_key = next((k for k in state.keys() if k.endswith("attn.rotary.cos")), None)
    head_dim = int(state[rotary_key].shape[-1]) if rotary_key is not None else 64
    n_head = max(1, n_embd // max(1, head_dim))

    mask_key = next((k for k in state.keys() if k.endswith("attn.mask")), None)
    context_size = int(state[mask_key].shape[-1]) if mask_key is not None else 1024

    return RyuuGPTConfig(
        vocab_size=vocab_size,
        context_size=context_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        use_reasoning_head=any(k.startswith("reasoning_head.") for k in state.keys()),
        use_value_head=any(k.startswith("value_head.") for k in state.keys()),
    )


def main():
    ckpts = candidate_checkpoints()
    if not ckpts:
        raise FileNotFoundError("No candidate checkpoints found under checkpoints/")

    tokenizer = BPETokenizer.load(str(ROOT / "tokenizer" / "bpe_tokenizer_postproc.json"))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("[INFO] Checkpoint matrix:")
    for p in ckpts:
        print(f"  - {p}")

    loaded = 0
    for p in ckpts:
        ckpt = torch.load(str(p), map_location=device)
        state = extract_state_dict(ckpt)

        cfg = None
        if isinstance(ckpt, dict):
            cfg_dict = ckpt.get("config") or ckpt.get("model_config")
            if cfg_dict:
                cfg = RyuuGPTConfig(**cfg_dict)
        if cfg is None:
            cfg = infer_cfg_from_state(state, tokenizer.vocab_size)

        model = RyuuGPT(cfg).to(device)
        model.load_state_dict(state, strict=False)
        model.eval()

        # Minimal forward sanity to prove runtime compatibility
        seq_len = min(8, cfg.context_size)
        x = torch.randint(0, cfg.vocab_size, (1, seq_len), device=device)
        with torch.no_grad():
            logits, _, _, _ = model(x)
        assert logits.shape[:2] == x.shape, f"Unexpected logits shape for {p}"

        loaded += 1
        print(f"[OK] Loaded + forward: {p}")

    print(f"[OK] Checkpoint matrix test passed ({loaded} checkpoints)")


if __name__ == "__main__":
    main()
