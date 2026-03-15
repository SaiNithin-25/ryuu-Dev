"""Train BPE tokenizer and save tokenizer/bpe_tokenizer_postproc.json."""

from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing

from data_pipeline.common import load_config


def iterate_texts(path):
    import json
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            p = obj.get("prompt", "")
            r = obj.get("response", "")
            if p and r:
                yield f"<|user|>\n{p}\n<|endofturn|>\n<|assistant|>\n{r}"


def main():
    cfg = load_config()
    train_jsonl = cfg["output"]["train"]
    out_path = Path(cfg["output"]["tokenizer"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tok_cfg = cfg["tokenizer"]
    special_tokens = tok_cfg["special_tokens"]

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)

    trainer = BpeTrainer(
        vocab_size=tok_cfg["vocab_size"],
        min_frequency=tok_cfg["min_frequency"],
        special_tokens=special_tokens,
        show_progress=True,
    )

    # Stream iterator to keep memory stable for large corpora.
    sample_count = 0
    for _ in iterate_texts(train_jsonl):
        sample_count += 1
        if sample_count > 0:
            break
    if sample_count == 0:
        raise ValueError("No training texts found for tokenizer")

    tokenizer.train_from_iterator(iterate_texts(train_jsonl), trainer=trainer)

    bos_id = tokenizer.token_to_id("<s>")
    eos_id = tokenizer.token_to_id("</s>")
    if bos_id is None or eos_id is None:
        raise ValueError("Tokenizer special tokens <s> and </s> were not learned correctly")
    tokenizer.post_processor = TemplateProcessing(
        single="<s> $A </s>",
        pair="<s> $A </s> <s> $B </s>",
        special_tokens=[("<s>", bos_id), ("</s>", eos_id)],
    )

    tokenizer.save(str(out_path))
    print(f"[OK] Tokenizer saved: {out_path}")
    print(f"[OK] Vocab size    : {tokenizer.get_vocab_size()}")
    print(f"[OK] Tokenizer mtime: {out_path.stat().st_mtime}")


if __name__ == "__main__":
    main()
