"""Build DPO seed pairs from train split with stronger rejected sampling."""

import json
import random
from data_pipeline.common import load_config, read_jsonl, ensure_parent, basic_token_count


def make_rejected(text):
    # Create a weaker local variant by truncation and cleanup.
    words = text.split()
    if len(words) > 20:
        cut = max(8, int(len(words) * 0.45))
        weak = " ".join(words[:cut])
    else:
        weak = text

    weak = weak.replace("\n", " ")
    weak = weak.replace("```", "")
    weak = weak[: max(32, int(len(weak) * 0.7))]
    return weak.strip()


def main():
    cfg = load_config()
    in_path = cfg["output"]["train"]
    out_path = cfg["output"]["dpo_seed"]
    dpo_cfg = cfg.get("dpo", {})
    max_pairs = int(dpo_cfg.get("max_pairs", 200_000))
    min_chosen_tokens = int(dpo_cfg.get("min_chosen_tokens", 20))
    min_rejected_tokens = int(dpo_cfg.get("min_rejected_tokens", 12))
    swap_probability = float(dpo_cfg.get("swap_probability", 0.65))

    rows = list(read_jsonl(in_path))
    random.seed(cfg["seed"])

    prompts = []
    chosen = []
    for row in rows:
        p = row.get("prompt", "").strip()
        c = row.get("response", "").strip()
        if not p or not c:
            continue
        if basic_token_count(c) < min_chosen_tokens:
            continue
        prompts.append(p)
        chosen.append(c)

    if not prompts:
        raise ValueError("No usable train rows found for DPO seed generation")

    ensure_parent(out_path)
    written = 0
    rejected_swapped = 0
    rejected_perturbed = 0

    with open(out_path, "w", encoding="utf-8") as f:
        n = len(prompts)
        for i in range(n):
            p = prompts[i]
            c = chosen[i]

            # Prefer swapped rejected answers from other prompts.
            if random.random() < swap_probability and n > 1:
                j = random.randrange(0, n - 1)
                if j >= i:
                    j += 1
                r = chosen[j]
                if r != c and basic_token_count(r) >= min_rejected_tokens:
                    rejected_swapped += 1
                else:
                    r = make_rejected(c)
                    rejected_perturbed += 1
            else:
                r = make_rejected(c)
                rejected_perturbed += 1

            if basic_token_count(r) < min_rejected_tokens or r == c:
                continue

            rec = {"prompt": p, "chosen": c, "rejected": r}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1
            if written >= max_pairs:
                break

    print(f"[OK] DPO seed written: {out_path}")
    print(f"[OK] Pairs count         : {written}")
    print(f"[OK] Rejected swapped    : {rejected_swapped}")
    print(f"[OK] Rejected perturbed  : {rejected_perturbed}")


if __name__ == "__main__":
    main()
