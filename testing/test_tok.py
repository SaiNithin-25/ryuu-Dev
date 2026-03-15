import sys
import os

# Ensure Windows console can print tokenizer output safely.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Add project root to path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from utils.bpe_tokenizer_v2 import BPETokenizer

# Load tokenizer (use correct filename: bpe_tokenizer_postproc.json)
tok = BPETokenizer.load("tokenizer/bpe_tokenizer_postproc.json")

print("[OK] Tokenizer loaded successfully")
print(f"Vocabulary size: {tok.vocab_size}")

sample = tok.encode("tokenizer behavior test")

print("\nEncoding test:")
print(f"  TYPE: {type(sample)}")
print(f"  VALUE: {sample}")

if hasattr(sample, "ids"):
    ids = sample.ids
else:
    ids = sample

print(f"  IDS: {ids}")
print(f"  DECODED: {tok.decode(ids)}")
