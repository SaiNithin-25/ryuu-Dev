"""Deduplicate prompt-response pairs using canonical hashes."""

import json
from hashlib import sha1
from data_pipeline.common import load_config, read_jsonl, canonicalize_text


def row_hash(row):
    key = canonicalize_text(row["prompt"]) + "\n" + canonicalize_text(row["response"])
    return sha1(key.encode("utf-8")).hexdigest()


def main():
    cfg = load_config()
    in_path = cfg["output"]["filtered"]
    out_path = cfg["output"]["deduped"]

    seen_pair = set()
    seen_prompt = set()
    seen_response = set()
    dup_pair = 0
    dup_prompt = 0
    dup_response = 0
    kept = 0

    with open(out_path, "w", encoding="utf-8") as out:
        for row in read_jsonl(in_path):
            pair_h = row_hash(row)
            p_h = sha1(canonicalize_text(row["prompt"]).encode("utf-8")).hexdigest()
            r_h = sha1(canonicalize_text(row["response"]).encode("utf-8")).hexdigest()

            if pair_h in seen_pair:
                dup_pair += 1
                continue
            if p_h in seen_prompt:
                dup_prompt += 1
                continue
            if r_h in seen_response:
                dup_response += 1
                continue

            seen_pair.add(pair_h)
            seen_prompt.add(p_h)
            seen_response.add(r_h)
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1

    print(f"[OK] Deduped rows: {kept}")
    print(f"[OK] Removed pair dups    : {dup_pair}")
    print(f"[OK] Removed prompt dups  : {dup_prompt}")
    print(f"[OK] Removed response dups: {dup_response}")
    print(f"[OK] Output      : {out_path}")


if __name__ == "__main__":
    main()
