# utils/patch_tokenizer_postproc.py
"""
Patch tokenizer/bpe_tokenizer.json to add TemplateProcessing that preserves special tokens.
Saves patched tokenizer to tokenizer/bpe_tokenizer_postproc.json and runs quick tests.
"""

import os, json, logging
from tokenizers import Tokenizer
from tokenizers.processors import TemplateProcessing

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("patch_tokenizer")

SRC = os.path.join("tokenizer", "bpe_tokenizer.json")
OUT = os.path.join("tokenizer", "bpe_tokenizer_postproc.json")

def load_tokenizer(path):
    tok = Tokenizer.from_file(path)
    return tok

def build_specials_list_from_json(path):
    """Return list of (string, id) special tokens found in JSON 'added_tokens' or 'special_tokens_map'."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    specials = []
    # added_tokens array structure (id, content) in your JSON
    for t in data.get("added_tokens", []):
        cid = t.get("id")
        content = t.get("content")
        if cid is not None and content:
            specials.append((content, int(cid)))
    # fallback: some tokenizers have special_tokens_map or added_tokens as dict
    if not specials:
        stm = data.get("special_tokens_map", {}) or data.get("special_tokens", {})
        for k, v in stm.items():
            # v might be dict or id
            if isinstance(v, dict):
                content = v.get("content") or v.get("token")
                tid = v.get("id")
            else:
                # nothing to do
                continue
            if content and tid is not None:
                specials.append((content, int(tid)))
    return specials

def apply_postprocessor(tok, specials):
    """Set TemplateProcessing preserving special tokens."""
    # Build minimal template: single = "$A", pair = "$A $B"
    # TemplateProcessing takes special_tokens list of (str, id)
    try:
        tok.post_processor = TemplateProcessing(
            single="$A",
            pair="$A $B",
            special_tokens=specials
        )
        return True
    except Exception as e:
        logger.exception("Failed to set TemplateProcessing: %s", e)
        return False

def run_roundtrip_tests(tok):
    tests = [
        "<|user|> Hello, how are you? <|assistant|>",
        "def example_fn(x):\n    return x * 2",
    ]
    for i, text in enumerate(tests, 1):
        enc = tok.encode(text)
        dec = tok.decode(enc.ids)
        ok = dec.strip() == text.strip()
        logger.info("[TEST %d] input=%r -> ids=%s roundtrip_ok=%s", i, text, enc.ids[:10], ok)
        logger.info(" encoded tokens sample: %s", enc.tokens[:10])
        logger.info(" decoded: %r", dec)

def main():
    if not os.path.exists(SRC):
        logger.error("Source tokenizer not found: %s", SRC)
        return 1

    specials = build_specials_list_from_json(SRC)
    logger.info("Found special tokens: %s", specials)

    tok = load_tokenizer(SRC)
    success = apply_postprocessor(tok, specials)
    if not success:
        logger.error("Could not apply post-processor. Exiting.")
        return 2

    # Save patched tokenizer
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tok.save(OUT)
    logger.info("Saved patched tokenizer to %s", OUT)

    # Run tests
    logger.info("Running roundtrip tests on patched tokenizer...")
    run_roundtrip_tests(tok)
    logger.info("Done.")

if __name__ == "__main__":
    main()
