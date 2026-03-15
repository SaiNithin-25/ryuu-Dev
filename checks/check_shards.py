import os
import sys
import torch
import glob

# Make project root importable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

SHARDS_DIR = os.path.join(ROOT, "data", "shards")


def check_shards():
    print("Checking shard files...")

    # Check if shards directory exists
    if not os.path.exists(SHARDS_DIR):
        print(f"[ERR] Shards directory not found: {SHARDS_DIR}")
        return

    # Look for train and val shards
    train_shards = sorted(glob.glob(os.path.join(SHARDS_DIR, "train_*.pt")))
    val_shards = sorted(glob.glob(os.path.join(SHARDS_DIR, "val_*.pt")))

    print(f"\nFound {len(train_shards)} training shards:")
    total_train_tokens = 0
    for shard_path in train_shards:
        try:
            shard_data = torch.load(shard_path, map_location="cpu")
            num_tokens = len(shard_data)
            size_mb = os.path.getsize(shard_path) / (1024 * 1024)
            print(f"[OK] {os.path.basename(shard_path)}: {num_tokens:,} tokens, {size_mb:.2f}MB")
            total_train_tokens += num_tokens
        except Exception as e:
            print(f"[ERR] Error loading {os.path.basename(shard_path)}: {str(e)}")

    print(f"\nFound {len(val_shards)} validation shards:")
    total_val_tokens = 0
    for shard_path in val_shards:
        try:
            shard_data = torch.load(shard_path, map_location="cpu")
            num_tokens = len(shard_data)
            size_mb = os.path.getsize(shard_path) / (1024 * 1024)
            print(f"[OK] {os.path.basename(shard_path)}: {num_tokens:,} tokens, {size_mb:.2f}MB")
            total_val_tokens += num_tokens
        except Exception as e:
            print(f"[ERR] Error loading {os.path.basename(shard_path)}: {str(e)}")

    print("\nSummary:")
    print(f"Total training tokens: {total_train_tokens:,}")
    print(f"Total validation tokens: {total_val_tokens:,}")
    print(f"Total tokens: {total_train_tokens + total_val_tokens:,}")


if __name__ == "__main__":
    check_shards()
