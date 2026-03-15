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


def resolve_checkpoint(prefer_dir: str | None = None) -> str:
    candidates = []
    if prefer_dir:
        candidates.extend([
            Path(prefer_dir) / "ckpt_best.pt",
        ])
        step_ckpts = sorted(glob.glob(str(Path(prefer_dir) / "ckpt_step*.pt")))
        if step_ckpts:
            candidates.append(Path(max(step_ckpts, key=lambda p: int(os.path.basename(p).split("step")[-1].split(".")[0]))))

    candidates.extend([
        ROOT / "checkpoints" / "v3_reasoning" / "ckpt_best.pt",
        ROOT / "checkpoints" / "phase5_long_baseline" / "ckpt_best.pt",
        ROOT / "checkpoints" / "phase3_train_smoke" / "ckpt_best.pt",
        ROOT / "checkpoints" / "test_smoke" / "ckpt_best.pt",
    ])

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


def safe_text(text: str) -> str:
    return text.encode("ascii", "replace").decode("ascii")


def run_query(model, tokenizer, device, cfg, user_prompt: str, max_new_tokens: int = 120):
    prompt = build_ryuu_dev_prompt(user_prompt)
    input_ids = torch.tensor([tokenizer.encode_ids(prompt)], dtype=torch.long, device=device)
    keep = min(input_ids.shape[1], max(16, cfg.context_size // 2))
    input_ids = input_ids[:, -keep:]

    eos_id = tokenizer.token_to_id("<|endofturn|>")
    out = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_k=40,
        top_p=0.9,
        do_sample=True,
        eos_token_id=eos_id,
    )

    decoded = tokenizer.decode(out[0].tolist())
    prefix_text = tokenizer.decode(input_ids[0].tolist())
    response = decoded[len(prefix_text):].strip() if decoded.startswith(prefix_text) else decoded

    return {
        "input_tokens": int(input_ids.shape[1]),
        "output_tokens": int(out.shape[1]),
        "response": safe_text(response),
    }


def main():
    import argparse

    parser = argparse.ArgumentParser("Run inference from latest checkpoint")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/v3_reasoning")
    parser.add_argument("--tokenizer", type=str, default="tokenizer/bpe_tokenizer_postproc.json")
    parser.add_argument("--prompt", type=str, default="Write a Python function to check if a list is sorted.")
    parser.add_argument("--max_new_tokens", type=int, default=120)
    args = parser.parse_args()

    try:
        ckpt_path = resolve_checkpoint(args.checkpoint_dir)
    except FileNotFoundError:
        print("[WAIT] No checkpoint found yet.")
        print("Expected after first eval/save interval (for your run this is step 1000).")
        print("Rerun this command once ckpt_step*.pt appears in the checkpoint dir.")
        return
    tokenizer = BPETokenizer.load(str(ROOT / args.tokenizer))
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

    result = run_query(model, tokenizer, device, cfg, args.prompt, args.max_new_tokens)

    print("[OK] Latest checkpoint inference complete")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Device: {device}")
    print(f"Input tokens: {result['input_tokens']}")
    print(f"Output tokens: {result['output_tokens']}")
    print("Response:")
    print(result["response"])


if __name__ == "__main__":
    main()
