import json

input_file = "data/raw/ryuu_prompts.jsonl"
output_file = "data/raw/ryuu_prompts_fixed.jsonl"

with open(input_file, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
    for line in fin:
        if not line.strip():
            continue
        obj = json.loads(line)
        # Rename fields if necessary
        obj["instruction"] = obj.get("prompt", obj.get("instruction", ""))
        obj["output"] = obj.get("completion", obj.get("output", ""))
        fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
print(f"✅ Converted prompts saved to {output_file}")