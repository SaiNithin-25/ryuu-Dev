import sys
from pathlib import Path
import glob
import os
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from model.config import RyuuGPTConfig
from model.Ryuu_gpt import RyuuGPT
from utils.bpe_tokenizer_v2 import BPETokenizer

print("Loading tokenizer...")
tokenizer = BPETokenizer.load("tokenizer/bpe_tokenizer_postproc.json")
print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

print("\nCreating model with config:")
cfg = RyuuGPTConfig(vocab_size=tokenizer.vocab_size)
print(f"  n_layer: {cfg.n_layer}")
print(f"  n_head: {cfg.n_head}")
print(f"  n_embd: {cfg.n_embd}")
print(f"  head_dim: {cfg.n_embd // cfg.n_head}")
print(f"  use_rope: {cfg.use_rope}")

model = RyuuGPT(cfg)
print(f"\nModel created successfully")
print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")

print("\nChecking rotary embedding shape...")
for name, param in model.named_parameters():
    if 'rotary' in name:
        print(f"  {name}: {param.shape}")
        break

print("\nLoading checkpoint...")

def resolve_checkpoint():
    candidates = [
        "checkpoints/ckpt_best.pt",
        "checkpoints/test_smoke/ckpt_best.pt",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    step_ckpts = sorted(glob.glob("checkpoints/**/ckpt_step*.pt", recursive=True))
    if step_ckpts:
        return max(step_ckpts, key=lambda p: int(os.path.basename(p).split("step")[-1].split(".")[0]))
    return None


def extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for key in ("model", "model_state", "state_dict"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
    return ckpt


ckpt_path = resolve_checkpoint()
if not ckpt_path:
    raise FileNotFoundError("No checkpoint found under checkpoints/")

print(f"  Checkpoint: {ckpt_path}")
state = torch.load(ckpt_path, map_location="cpu")

# Check checkpoint structure
state_dict = extract_state_dict(state)
has_config = isinstance(state, dict) and ("config" in state or "model_config" in state)
print(f"  Has config key: {has_config}")

# Check checkpoint rotary shapes
print("\nCheckpoint rotary embeddings:")
for key in state_dict.keys():
    if 'rotary' in key:
        print(f"  {key}: {state_dict[key].shape}")
        if 'cos' in key:
            break

print("\nAttempting to load state_dict...")
model_state = model.state_dict()
compatible_state = {
    k: v for k, v in state_dict.items()
    if k in model_state and getattr(v, "shape", None) == model_state[k].shape
}

missing, unexpected = model.load_state_dict(compatible_state, strict=False)
print("Checkpoint loaded with shape-compatible keys only")
print(f"  Loaded keys: {len(compatible_state)} / {len(state_dict)}")
print(f"  Missing keys: {len(missing)}")
print(f"  Unexpected keys: {len(unexpected)}")
