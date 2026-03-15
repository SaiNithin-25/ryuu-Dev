import json
import random
import argparse
import time
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F

import os
import sys

# ---------------------------------------------------
# Project root path fix (CRITICAL)
# ---------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.append(ROOT)
from model.Ryuu_gpt import RyuuGPT
from model.config import RyuuGPTConfig
from utils.bpe_tokenizer_v2 import BPETokenizer

# --------------------------------------------------
# CLI
# --------------------------------------------------
parser = argparse.ArgumentParser("Generate DPO dataset (ULTRA-FAST - Batch + Short Tokens)")
parser.add_argument("--ckpt", type=str, required=True)
parser.add_argument("--tokenizer", type=str, required=True)
parser.add_argument("--out", type=str, default="data/dpo/ryuu_dpo.jsonl")
parser.add_argument("--num_samples", type=int, default=5000)
parser.add_argument("--max_new_tokens", type=int, default=64)  # REDUCED: 128 → 64 (2x faster)
parser.add_argument("--temperature", type=float, default=0.8)
parser.add_argument("--top_k", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=8)  # INCREASED: 4 → 8 (better GPU utilization)
args = parser.parse_args()

# --------------------------------------------------
# Device
# --------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# --------------------------------------------------
# Load tokenizer
# --------------------------------------------------
tokenizer = BPETokenizer.load(args.tokenizer)
EOS_ID = tokenizer.eos_token_id if hasattr(tokenizer, "eos_token_id") else tokenizer.encode("</s>").ids[0]
print(f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size}, EOS_ID: {EOS_ID}")

# --------------------------------------------------
# Load model
# --------------------------------------------------
cfg = RyuuGPTConfig(
    vocab_size=tokenizer.vocab_size,
    context_size=1024,
    n_layer=16,
    n_head=12,
    n_embd=768,
    use_reasoning_head=False,   # 🔒 hidden reasoning
)

model = RyuuGPT(cfg).to(device)
ckpt = torch.load(args.ckpt, map_location=device)

def _extract_state_dict(data):
    if isinstance(data, dict):
        for key in ("model", "model_state", "state_dict"):
            if key in data and isinstance(data[key], dict):
                return data[key]
    return data

state = _extract_state_dict(ckpt)
missing, unexpected = model.load_state_dict(state, strict=False)

print("Missing keys:", missing)
print("Ignored keys:", unexpected)
model.eval()

# --------------------------------------------------
# Expanded prompt pool (50 diverse prompts)
# --------------------------------------------------
PROMPTS = [
    # Programming
    "Explain how a binary search works.",
    "Write a Python function to reverse a list.",
    "How does recursion work in programming?",
    "Write a function to check if a string is a palindrome.",
    "Explain what a hash table is and when to use it.",
    "Write a quicksort algorithm in pseudocode.",
    "How does garbage collection work in Python?",
    "Explain the difference between lists and tuples.",
    "Write a function to find the longest substring without repeating characters.",
    "How does dynamic programming work?",
    
    # Machine Learning
    "What is overfitting in machine learning?",
    "Explain gradient descent simply.",
    "How does backpropagation work?",
    "Explain transformers in simple terms.",
    "What is the difference between supervised and unsupervised learning?",
    "Explain neural networks in simple terms.",
    "What is regularization and why is it important?",
    "How does cross-validation work?",
    "Explain the concept of activation functions.",
    "What is a convolutional neural network?",
    
    # Data Structures
    "Explain what a binary tree is and its properties.",
    "How does a linked list work?",
    "Explain the difference between stacks and queues.",
    "What is a graph and what are common graph algorithms?",
    "How does a red-black tree maintain balance?",
    "Explain what a heap is and its properties.",
    "What is a trie data structure and when to use it?",
    "Explain the concept of graph traversal methods.",
    
    # Advanced Topics
    "Write a short story about an AI assistant.",
    "How does encryption work at a basic level?",
    "Explain what distributed systems are.",
    "What is microservices architecture?",
    "How does caching improve performance?",
    "Explain what containerization is.",
    "How does load balancing work?",
    "What is API design and best practices?",
    "Explain the difference between SQL and NoSQL.",
    "What is machine learning model deployment?",
    
    # Problem-Solving
    "How would you solve the traveling salesman problem?",
    "Explain the difference between greedy and dynamic programming approaches.",
    "How would you debug a slow database query?",
    "What are common security vulnerabilities and how to prevent them?",
    "How would you design a URL shortener service?",
    "Explain what a software design pattern is with examples.",
    "How does version control work?",
    "What is the importance of code review?",
    "How would you optimize an image processing algorithm?",
    "Explain what refactoring is and why it matters.",
]

# --------------------------------------------------
# FAST Batch Generation Function
# --------------------------------------------------
@torch.no_grad()
def generate_batch(prompts, max_new=128, temperature=0.8, top_k=50):
    """
    Generate multiple prompts in parallel for speed.
    
    Args:
        prompts: list of prompt strings
        max_new: max new tokens
        temperature: sampling temperature
        top_k: top-k filtering
    
    Returns:
        list of generated text strings
    """
    model.eval()
    batch_size = len(prompts)
    
    # Encode all prompts (keep original index)
    encoded = []
    for i, prompt in enumerate(prompts):
        enc = tokenizer.encode(prompt)
        ids = enc.ids if hasattr(enc, "ids") else enc
        encoded.append((i, ids))

    # Group by length to avoid pad-token contamination
    buckets = {}
    for i, ids in encoded:
        buckets.setdefault(len(ids), []).append((i, ids))

    results = [None] * batch_size

    for _, items in buckets.items():
        # Build batch tensor with uniform length (no padding needed)
        idxs = [i for i, _ in items]
        ids_list = [ids for _, ids in items]
        x = torch.tensor(ids_list, dtype=torch.long, device=device)
        seq_len = x.shape[1]

        max_len = min(seq_len + max_new, 2048)
        output = torch.full((len(items), max_len), EOS_ID, dtype=torch.long, device=device)
        output[:, :seq_len] = x

        current_len = seq_len

        for _ in range(max_new):
            logits, _, _, _ = model(output[:, :current_len])

            logits = logits[:, -1, :] / max(temperature, 0.1)
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)

            output[:, current_len] = next_ids
            current_len += 1

            if (next_ids == EOS_ID).all():
                break

        for row, orig_i in enumerate(idxs):
            generated_ids = output[row, :current_len].tolist()
            text = tokenizer.decode(generated_ids) if hasattr(tokenizer, "decode") else "<no-decode>"
            results[orig_i] = text

    return results


def score(text, prompt=""):
    """
    Improved scoring combining:
    - Length heuristics
    - Structure and clarity indicators
    - Vocabulary complexity
    - Reduced penalties to minimize ties
    """
    s = 0.0
    words = text.split()
    length = len(words)

    # Length scoring
    if 30 <= length <= 500:
        s += 1.0
    elif 15 <= length < 30:
        s += 0.5
    elif length > 500:
        s -= 0.3
    elif length < 15:
        s -= 0.5

    # Penalize extreme rambling
    if length > 800:
        s -= 1.0

    # Structure indicators
    if "\n" in text or ("-" in text and text.count("-") > 2):
        s += 0.3
    if ":" in text:
        s += 0.2

    # Vocabulary quality
    avg_word_len = sum(len(w) for w in words) / max(len(words), 1)
    if avg_word_len > 5:
        s += 0.2
    if avg_word_len > 6:
        s += 0.2

    # Penalize low-quality phrases
    bad_phrases = ["i don't know", "i'm not sure", "unclear"]
    count = sum(1 for phrase in bad_phrases if phrase in text.lower())
    s -= count * 0.3

    # Reward completeness indicators
    good_phrases = ["because", "therefore", "the reason", "for example", "in summary"]
    count = sum(1 for phrase in good_phrases if phrase in text.lower())
    s += count * 0.15

    # Tie-breaking randomness
    s += random.uniform(-0.05, 0.05)

    return s


def get_resume_state(output_path):
    """Check if output file exists and return number of samples already generated."""
    if not os.path.exists(output_path):
        return 0, None
    
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        num_generated = len(lines)
        print(f"📋 Found existing output with {num_generated} samples. Resuming from sample {num_generated}...")
        return num_generated, lines
    except Exception as e:
        print(f"⚠️  Error reading resume state: {e}. Starting fresh.")
        return 0, None


def save_checkpoint(checkpoint_path, stats, elapsed_time, step):
    """Save checkpoint for resuming later."""
    checkpoint = {
        "step": step,
        "stats": dict(stats),
        "elapsed_time": elapsed_time,
        "timestamp": time.time(),
    }
    try:
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(checkpoint, f, indent=2)
    except Exception as e:
        print(f"⚠️  Failed to save checkpoint: {e}")

# --------------------------------------------------
# Generate DPO pairs with BATCH mode
# --------------------------------------------------
Path(args.out).parent.mkdir(parents=True, exist_ok=True)

# Check for resume
resume_count, existing_lines = get_resume_state(args.out)
checkpoint_path = args.out.replace(".jsonl", ".checkpoint.json")

start_time = time.time()
stats = {
    "total_generated": resume_count,  # Start from existing count
    "total_skipped_ties": 0,
    "total_skipped_errors": 0,
    "generation_times": [],
    "prompt_usage": defaultdict(int),
}

# Append mode if resuming, write mode if fresh start
file_mode = "a" if resume_count > 0 else "w"

with open(args.out, file_mode, encoding="utf-8") as f:
    i = resume_count  # Start from resume point
    attempts = 0
    max_attempts = args.num_samples * 3
    
    while i < args.num_samples and attempts < max_attempts:
        try:
            # Select batch of prompts
            batch_prompts = [random.choice(PROMPTS) for _ in range(args.batch_size)]
            for p in batch_prompts:
                stats["prompt_usage"][p] += 1
            
            # Generate batch (2 sets for each prompt = 2 responses per prompt)
            gen_start = time.time()
            batch_a = generate_batch(batch_prompts, max_new=args.max_new_tokens, 
                                     temperature=args.temperature, top_k=args.top_k)
            batch_b = generate_batch(batch_prompts, max_new=args.max_new_tokens,
                                     temperature=args.temperature, top_k=args.top_k)
            gen_time = time.time() - gen_start
            stats["generation_times"].append(gen_time)
            
            # Process each prompt's pair
            for prompt, a, b in zip(batch_prompts, batch_a, batch_b):
                attempts += 1
                
                # Score
                sa = score(a, prompt)
                sb = score(b, prompt)
                
                # Skip if too similar or invalid
                if abs(sa - sb) < 0.1:
                    stats["total_skipped_ties"] += 1
                    continue
                
                if len(a.strip()) < 10 or len(b.strip()) < 10:
                    stats["total_skipped_errors"] += 1
                    continue
                
                # Write
                chosen, rejected = (a, b) if sa > sb else (b, a)
                record = {
                    "prompt": prompt,
                    "chosen": chosen.strip(),
                    "rejected": rejected.strip(),
                    "score_diff": abs(sa - sb),
                }
                
                f.write(json.dumps(record) + "\n")
                stats["total_generated"] += 1
                i += 1
                
                if i >= args.num_samples:
                    break
            
            # Progress report
            if i % 100 == 0 or i >= args.num_samples:
                elapsed = time.time() - start_time
                rate = (i - resume_count) / elapsed if elapsed > 0 else 0
                eta = (args.num_samples - i) / max(rate, 0.001)
                print(f"[{i}/{args.num_samples}] Rate: {rate:.2f} samples/sec | Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s")
                # Save checkpoint for resume capability
                save_checkpoint(checkpoint_path, stats, elapsed, i)
        
        except Exception as e:
            stats["total_skipped_errors"] += 1
            print(f"Error: {e}")
            continue

elapsed_time = time.time() - start_time

# Print statistics
print("\n" + "="*70)
print("✅ DPO dataset generation complete (BATCH MODE)")
print("="*70)
print(f"Saved to: {args.out}")
print(f"\nStatistics:")
print(f"  Total samples generated: {stats['total_generated']}")
if resume_count > 0:
    print(f"  (Resumed from {resume_count} previous samples)")
print(f"  Total attempts: {attempts}")
print(f"  Skipped (ties): {stats['total_skipped_ties']}")
print(f"  Skipped (errors): {stats['total_skipped_errors']}")
print(f"  Total time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
print(f"  Average time per sample: {elapsed_time / max(stats['total_generated'] - resume_count, 1):.3f} seconds")
print(f"  Generation rate: {(stats['total_generated'] - resume_count) / elapsed_time:.2f} samples/sec")
print(f"\nGeneration timings (batch of {args.batch_size}):")
if stats["generation_times"]:
    gen_times = stats["generation_times"]
    print(f"  Min: {min(gen_times):.3f}s | Max: {max(gen_times):.3f}s | Mean: {sum(gen_times)/len(gen_times):.3f}s")
print(f"\nTop prompts used:")
for prompt, count in sorted(stats["prompt_usage"].items(), key=lambda x: x[1], reverse=True)[:5]:
    print(f"  {count}x - {prompt[:50]}...")
