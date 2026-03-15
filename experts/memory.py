# experts/memory.py
from __future__ import annotations
import re
import math
import time
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

# ---------------------------
# Utils
# ---------------------------
_CODE_FENCE = re.compile(r"```([\s\S]*?)```", re.MULTILINE)
_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]+")

def _tokenize(text: str) -> List[str]:
    return [w.lower() for w in _WORD.findall(text)]

def _now() -> float:
    return time.time()

# ---------------------------
# Session memory (FIFO)
# ---------------------------
@dataclass
class Turn:
    role: str               # "user" | "assistant"
    text: str
    ts: float = field(default_factory=_now)

class SessionMemory:
    """
    Keeps last N turns (user + assistant). Exportable to a compact,
    model-friendly context string.
    """
    def __init__(self, max_turns: int = 16, max_chars: int = 16_000):
        self.max_turns = max_turns
        self.max_chars = max_chars
        self._turns: List[Turn] = []

    def add(self, role: str, text: str):
        self._turns.append(Turn(role, text))
        # trim by turns
        if len(self._turns) > self.max_turns:
            self._turns = self._turns[-self.max_turns:]
        # trim by chars
        total = 0
        pruned: List[Turn] = []
        for t in reversed(self._turns):
            total += len(t.text)
            pruned.append(t)
            if total >= self.max_chars:
                break
        self._turns = list(reversed(pruned))

    def export(self) -> str:
        # Compact, delimitered dialogue block
        lines = []
        for t in self._turns:
            header = "User:" if t.role == "user" else "Assistant:"
            lines.append(f"{header} {t.text}".strip())
        return "\n".join(lines)

    def clear(self):
        self._turns.clear()

# ---------------------------
# Snippet store (code/errors)
# ---------------------------
@dataclass
class Snippet:
    kind: str               # "code" | "error" | "note"
    text: str
    meta: Dict = field(default_factory=dict)
    ts: float = field(default_factory=_now)

class SnippetStore:
    """
    In-process store with a **fast TF-IDF-ish** scorer.
    Good enough for few-hundred/low-thousands snippets.
    """
    def __init__(self):
        self.snippets: List[Snippet] = []
        self.doc_freq: Dict[str, int] = {}  # idf cache (term -> doc count)

    def add(self, kind: str, text: str, **meta):
        text = text.strip()
        if not text:
            return
        self.snippets.append(Snippet(kind, text, meta))
        # update docfreqs once per snippet
        seen = set(_tokenize(text))
        for tok in seen:
            self.doc_freq[tok] = self.doc_freq.get(tok, 0) + 1

    def _score(self, query_tokens: List[str], snippet: Snippet) -> float:
        if not query_tokens:
            return 0.0
        toks = _tokenize(snippet.text)
        if not toks:
            return 0.0
        # term frequency
        tf: Dict[str, int] = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        # idf
        N = max(1, len(self.snippets))
        score = 0.0
        for q in query_tokens:
            df = self.doc_freq.get(q, 0)
            idf = math.log((N + 1) / (df + 1)) + 1.0
            score += (tf.get(q, 0)) * idf
        # recency bonus (soft)
        age_sec = max(1.0, _now() - snippet.ts)
        recency = 1.0 / (1.0 + math.log10(age_sec))
        return score * recency

    def search(self, query: str, top_k: int = 6, kinds: Optional[List[str]] = None) -> List[Tuple[Snippet, float]]:
        q = _tokenize(query)
        cand: List[Tuple[Snippet, float]] = []
        for s in self.snippets:
            if kinds and s.kind not in kinds:
                continue
            cand.append((s, self._score(q, s)))
        cand.sort(key=lambda x: x[1], reverse=True)
        return cand[:top_k]

# ---------------------------
# Context builder
# ---------------------------
class ContextBuilder:
    """
    Builds a compact context block for the model:
    - recent dialogue (SessionMemory)
    - top snippets (code/errors/notes)
    - optional structured “task hints”
    """
    def __init__(self, session: SessionMemory, store: SnippetStore,
                 max_ctx_chars: int = 4000,
                 max_snippets: int = 6):
        self.session = session
        self.store = store
        self.max_ctx_chars = max_ctx_chars
        self.max_snippets = max_snippets

    def build(self, user_query: str) -> str:
        # 1) find relevant snippets
        hits = self.store.search(user_query, top_k=self.max_snippets)

        blocks = []
        if hits:
            blocks.append("<<Relevant Snippets>>")
            for snip, score in hits:
                tag = snip.kind.upper()
                body = snip.text
                # collapse long code fences
                if len(body) > 1000:
                    body = body[:1000] + "\n... [truncated]\n"
                blocks.append(f"[{tag}] score={score:.2f}\n{body}")

        # 2) recent dialogue
        dialog = self.session.export()
        if dialog:
            blocks.append("<<Recent Dialogue>>")
            blocks.append(dialog)

        # trim to max chars
        ctx = "\n\n".join(blocks)
        if len(ctx) > self.max_ctx_chars:
            ctx = ctx[-self.max_ctx_chars:]
        return ctx

    # helpers to save common snippet types
    def add_code(self, text: str, **meta): self.store.add("code", self._ensure_fenced(text), **meta)
    def add_error(self, text: str, **meta): self.store.add("error", text, **meta)
    def add_note(self, text: str, **meta): self.store.add("note", text, **meta)

    @staticmethod
    def _ensure_fenced(text: str) -> str:
        if _CODE_FENCE.search(text):
            return text
        return f"```python\n{text.strip()}\n```"
