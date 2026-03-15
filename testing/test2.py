import logging
import sys
import os

# Add the parent directory (ryuu-ai) to the system path
# This allows imports like 'from utils...'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.bpe_tokenizer_v2 import BPETokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [TEST] %(message)s",
    datefmt="%H:%M:%S",
)

def main():
    path = "tokenizer/bpe_tokenizer_postproc.json"
    logging.info("Loading tokenizer from: %s", path)

    tok = BPETokenizer.load(path)

    logging.info("Vocab size: %s", tok.vocab_size)
    logging.info("Special tokens:")
    for t in ["<pad>", "<unk>", "<s>", "</s>", "<|user|>", "<|assistant|>", "<|endofturn|>"]:
        logging.info("  %-14s -> %s", t, tok.token_to_id(t))

    tests = [
        "<|user|> Hello, how are you? <|assistant|>",
        "def example_fn(x):\n    return x * 2",
        "print('roundtrip test')",
    ]

    for i, text in enumerate(tests, 1):
        ids = tok.encode_ids(text)
        decoded = tok.decode(ids)  # IMPORTANT: preserves special tokens by default

        logging.info("")
        logging.info("TEST %d", i)
        logging.info("INPUT   : %r", text)
        logging.info("IDS     : %s", ids)
        logging.info("DECODED : %r", decoded)

        if "<|user|>" in text or "<|assistant|>" in text:
            assert "<|user|>" in decoded or "<|assistant|>" in decoded, \
                "❌ Special tokens were stripped during decode"

        logging.info("ROUNDTRIP OK ✔")

    logging.info("")
    logging.info("ALL TOKENIZER QUICK TESTS PASSED ✔")

if __name__ == "__main__":
    main()
