# quick smoke test
import sys
import os
import glob
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experts.reasoning import RyuuDevReasoner


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
    raise FileNotFoundError("No checkpoint found under checkpoints/")

def safe_print(text):
    print(str(text).encode("ascii", "replace").decode("ascii"))


reasoner = RyuuDevReasoner(
    model_ckpt=resolve_checkpoint(),
    tokenizer_path="tokenizer/bpe_tokenizer_postproc.json",
)

# turn 1: generate code
r1 = reasoner.run("Write a Python function to check if a string is a palindrome, with tests.", max_tokens=200)
safe_print(r1["output"])

# turn 2: simulate an error
err = """Traceback (most recent call last):
  File "main.py", line 3, in <module>
    assert is_pal('Abc')
AssertionError"""
r2 = reasoner.run(f"I'm getting this error:\n{err}\nPlease debug.", max_tokens=200)
safe_print(r2["output"])

# turn 3: ask for explanation (will use snippets + history)
r3 = reasoner.run("Explain why my first attempt failed and how your fix addresses it.", max_tokens=200)
safe_print(r3["output"])
print("[OK] Reasoning test completed.")
