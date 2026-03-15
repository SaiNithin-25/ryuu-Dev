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
from Inference.prompt_builder import build_ryuu_dev_prompt


def resolve_checkpoint() -> str:
    candidates = [
        ROOT / "checkpoints" / "ckpt_best.pt",
        ROOT / "checkpoints" / "test_smoke" / "ckpt_best.pt",
        ROOT / "checkpoints" / "phase3_train_smoke" / "ckpt_best.pt",
    ]
    for p in candidates:
        if p.exists():
            return str(p)

    step_ckpts = glob.glob(str(ROOT / "checkpoints" / "**" / "ckpt_step*.pt"), recursive=True)
    if step_ckpts:
        return max(step_ckpts, key=lambda p: int(os.path.basename(p).split("step")[-1].split(".")[0]))
    raise FileNotFoundError("No checkpoint found under checkpoints/")


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


def load_model_and_tokenizer():
    ckpt_path = resolve_checkpoint()
    tokenizer = BPETokenizer.load(str(ROOT / "tokenizer" / "bpe_tokenizer_postproc.json"))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(ckpt_path, map_location=device)
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
    return model, tokenizer, device, cfg, ckpt_path


def main():
    torch.manual_seed(123)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(123)

    model, tokenizer, device, cfg, ckpt_path = load_model_and_tokenizer()

    prompt = build_ryuu_dev_prompt("Return exactly one short Python line that squares x.")
    input_ids = torch.tensor([tokenizer.encode_ids(prompt)], dtype=torch.long, device=device)
    keep = min(input_ids.shape[1], max(8, cfg.context_size // 2))
    input_ids = input_ids[:, -keep:]

    eos_id = tokenizer.token_to_id("<|endofturn|>")

    out1 = model.generate(
        input_ids.clone(),
        max_new_tokens=24,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        do_sample=False,
        eos_token_id=eos_id,
    )
    out2 = model.generate(
        input_ids.clone(),
        max_new_tokens=24,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        do_sample=False,
        eos_token_id=eos_id,
    )

    assert torch.equal(out1, out2), "Deterministic generation mismatch between two runs"
    assert out1.shape[1] > input_ids.shape[1], "No new tokens were generated in deterministic mode"

    txt = tokenizer.decode(out1[0].tolist())
    txt = txt.encode("ascii", "replace").decode("ascii")

    print("[OK] Deterministic generation passed")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Device: {device}")
    print(f"Input tokens: {input_ids.shape[1]}")
    print(f"Output tokens: {out1.shape[1]}")
    print("Sample output:")
    print(txt[:240])


if __name__ == "__main__":
    main()
