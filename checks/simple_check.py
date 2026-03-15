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

# Just check the checkpoint
ckpt = torch.load(ckpt_path, map_location="cpu")
print("Checkpoint structure:")
print(f"Type: {type(ckpt)}")
print(f"Keys: {list(ckpt.keys()) if isinstance(ckpt, dict) else 'Not a dict'}")

state = extract_state_dict(ckpt)

# Count blocks
blocks = sorted(set(int(k.split('.')[1]) for k in state.keys() if k.startswith('blocks.')))
print(f"\nBlocks in checkpoint: {blocks}")
print(f"Number of layers: {len(blocks)}")

# Check rotary shapes
rot_cos = [k for k in state.keys() if 'attn.rotary.cos' in k]
if rot_cos:
    key = rot_cos[0]
    print(f"\nRotary cos shape: {state[key].shape}")
    print(f"Head dimension (from rotary): {state[key].shape[-1]}")
