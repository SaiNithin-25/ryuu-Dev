import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.bpe_tokenizer_v2 import BPETokenizer


def main():
    tok = BPETokenizer.load(str(ROOT / "tokenizer" / "bpe_tokenizer_postproc.json"))

    specials = ["<pad>", "<unk>", "<s>", "</s>", "<|user|>", "<|assistant|>", "<|endofturn|>"]
    ids = {}
    for t in specials:
        tid = tok.token_to_id(t)
        assert tid is not None, f"Missing special token id: {t}"
        ids[t] = tid

    sample = "<|user|> hello </s> <|assistant|>"
    enc = tok.encode_ids(sample)
    dec = tok.decode(enc)

    assert ids["<|user|>"] in enc, "Encoded ids missing <|user|>"
    assert ids["<|assistant|>"] in enc, "Encoded ids missing <|assistant|>"
    assert ids["</s>"] in enc, "Encoded ids missing </s>"
    assert "<|user|>" in dec, "Decoded text dropped <|user|>"
    assert "<|assistant|>" in dec, "Decoded text dropped <|assistant|>"

    no_skip = tok.decode(enc, skip_special_tokens=False)
    with_skip = tok.decode(enc, skip_special_tokens=True)
    assert len(with_skip) <= len(no_skip), "skip_special_tokens=True did not reduce/same text length"

    eos_id = ids["</s>"]
    eos_text = tok.decode([eos_id])
    assert "</s>" in eos_text or eos_text.strip() == "", "EOS decode behavior unexpected"

    print("[OK] Tokenizer consistency passed")
    print(f"Vocab size: {tok.vocab_size}")
    print("Special IDs:")
    for t in specials:
        print(f"  {t}: {ids[t]}")
    print(f"Roundtrip sample: {dec.encode('ascii', 'replace').decode('ascii')}")


if __name__ == "__main__":
    main()
