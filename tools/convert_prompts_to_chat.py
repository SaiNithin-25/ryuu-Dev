import json
import re

input_path = "data/raw/ryuu_prompts.jsonl"
output_path = "data/raw/ryuu_chat_format.jsonl"

def escape_invalid_json(line):
    # Escape unescaped backslashes
    return re.sub(r'(?<!\\)\\(?![\\ntr"])', r'\\\\', line)

with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
    for line in infile:
        line = line.strip()
        try:
            # Try parsing directly
            obj = json.loads(line)
        except json.JSONDecodeError:
            # Try escaping invalid backslashes
            try:
                fixed_line = escape_invalid_json(line)
                obj = json.loads(fixed_line)
            except Exception as e:
                print(f"❌ Still invalid after fix: {e}")
                continue

        chat_format = {
            "messages": [
                {"role": "user", "content": obj["prompt"]},
                {"role": "assistant", "content": obj["completion"]}
            ]
        }
        outfile.write(json.dumps(chat_format, ensure_ascii=False) + "\n")

print(f"✅ All valid prompts converted to chat format → {output_path}")
