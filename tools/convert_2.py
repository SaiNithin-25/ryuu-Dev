import json
import os

# Correct paths — no double ".json"
input_file = "data/clean/python100k_train.json"   # ✅ this file exists in your folder
output_file = "data/clean/python100k_train_fixed.jsonl"

if not os.path.exists(input_file):
    raise FileNotFoundError(f"❌ Input file not found: {input_file}")

# Read JSONL format safely
with open(input_file, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f if line.strip()]

# Write normalized file
with open(output_file, "w", encoding="utf-8") as f:
    for d in data:
        instruction = d.get("prompt") or d.get("instruction") or ""
        output = d.get("completion") or d.get("output") or ""
        f.write(json.dumps({"instruction": instruction, "output": output}, ensure_ascii=False) + "\n")

print(f"✅ Fixed file saved to {output_file}")
