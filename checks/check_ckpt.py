import glob
import os
import torch


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

print("Checkpoint:", ckpt_path)
ckpt = torch.load(ckpt_path, map_location="cpu")
print("Top-level keys:", list(ckpt.keys())[:5] if isinstance(ckpt, dict) else "raw state_dict")

# Determine if this is a full checkpoint or just state_dict
state = extract_state_dict(ckpt)
print("State format resolved")

# Find blocks
blocks = [k for k in state.keys() if k.startswith('blocks.')]
print(f"Number of block keys: {len(blocks)}")

if blocks:
    block_indices = [int(k.split('.')[1]) for k in blocks]
    max_block = max(block_indices)
    print(f"Max block index: {max_block}")
    print(f"Total blocks in checkpoint: {max_block + 1}")
    
# Check rotary dimensions
rotary_keys = [k for k in state.keys() if 'rotary' in k]
if rotary_keys:
    print(f"\nFirst rotary key: {rotary_keys[0]}")
    print(f"Shape: {state[rotary_keys[0]].shape}")
