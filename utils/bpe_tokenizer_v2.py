# utils/bpe_tokenizer_v2.py
"""
BPETokenizer wrapper for tokenizers.Tokenizer that preserves special tokens
on decode by default (skip_special_tokens=False).

Usage:
    from utils.bpe_tokenizer_v2 import BPETokenizer
    tokenizer = BPETokenizer.load("tokenizer/bpe_tokenizer_postproc.json")
    ids = tokenizer.encode_ids("<|user|> Hello")
    text = tokenizer.decode(ids)  # preserves special tokens by default
"""

from typing import List, Optional, Union
from tokenizers import Tokenizer
import json
import os

class BPETokenizer:
    def __init__(self, path_or_tokenizer: Union[str, Tokenizer]):
        if isinstance(path_or_tokenizer, Tokenizer):
            self.tok = path_or_tokenizer
            self._path = None
        else:
            self._path = path_or_tokenizer
            self.tok = Tokenizer.from_file(path_or_tokenizer)

        # Build special id -> token map (best-effort)
        self._special_id_to_token = {}
        try:
            # tokenizers exposes added_tokens via get_vocab or .added_tokens
            added = getattr(self.tok, "added_tokens", None)
            if added:
                for a in added:
                    # a may be a dict or object
                    try:
                        tid = a["id"]
                        content = a["content"]
                    except Exception:
                        try:
                            tid = a.id
                            content = a.content
                        except Exception:
                            continue
                    if tid is not None and content is not None:
                        self._special_id_to_token[int(tid)] = content
        except Exception:
            # ignore if not available
            pass

    @classmethod
    def load(cls, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Tokenizer file not found: {path}")
        return cls(path)

    def save(self, path: str):
        """Save the underlying tokenizer JSON to path."""
        # tokenizers.Tokenizer has .save()
        return self.tok.save(path)

    def encode(self, text: str):
        """Return the Encoding object from tokenizers (has .ids, .tokens)."""
        return self.tok.encode(text)

    def encode_ids(self, text: str) -> List[int]:
        """Return list of token ids for the input text."""
        return self.tok.encode(text).ids

    def decode(self, ids: List[int], skip_special_tokens: Optional[bool] = None) -> str:
        """
        Decode ids -> text.
        By default we preserve special tokens (skip_special_tokens=False) to keep markers.
        If explicit skip_special_tokens=True passed, we respect that.
        """
        if skip_special_tokens is None:
            skip_special_tokens = False

        # Newer tokenizers support skip_special_tokens kwarg
        try:
            return self.tok.decode(ids, skip_special_tokens=skip_special_tokens)
        except TypeError:
            # Fallback: older tokenizers may not accept the kwarg
            txt = self.tok.decode(ids)
            if skip_special_tokens:
                # Best-effort removal of special tokens by string replace
                for tok in set(self._special_id_to_token.values()):
                    txt = txt.replace(tok, "")
            return txt

    @property
    def vocab_size(self) -> int:
        # try multiple ways to get vocab size
        try:
            return int(self.tok.get_vocab_size())
        except Exception:
            try:
                return int(getattr(self.tok, "vocab_size"))
            except Exception:
                return -1

    def token_to_id(self, token: str) -> Optional[int]:
        """Return token id for a token string, or None if missing."""
        try:
            return self.tok.token_to_id(token)
        except Exception:
            try:
                v = self.tok.get_vocab()
                return v.get(token)
            except Exception:
                return None

    def id_to_token(self, idx: int) -> Optional[str]:
        try:
            # prefer explicit special map
            if int(idx) in self._special_id_to_token:
                return self._special_id_to_token[int(idx)]
        except Exception:
            pass
        try:
            return self.tok.id_to_token(int(idx))
        except Exception:
            return None

    # Convenience: allow using the wrapper like the original one in some places
    def __getattr__(self, name):
        # Delegate unknown attributes to underlying tokenizer where reasonable
        if hasattr(self.tok, name):
            return getattr(self.tok, name)
        raise AttributeError(f"{self.__class__.__name__} has no attribute {name}")
