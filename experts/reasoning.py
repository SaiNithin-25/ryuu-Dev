# experts/ryuu_dev_reasoning.py
import re
import os
import glob
import torch
import logging
from typing import Optional
from experts.memory import SessionMemory, SnippetStore, ContextBuilder
from model.Ryuu_gpt import RyuuGPT
from model.config import RyuuGPTConfig
from utils.bpe_tokenizer_v2 import BPETokenizer

CODE_FENCE = re.compile(r"```([\s\S]*?)```", re.MULTILINE)
ERROR_HINT = re.compile(r"(Traceback|Error:|Exception:|RuntimeError|TypeError|ValueError|ReferenceError)", re.IGNORECASE)


def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for key in ("model", "model_state", "state_dict"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
    return ckpt


def _resolve_checkpoint(path: str) -> str:
    if os.path.isfile(path):
        return path
    if os.path.isdir(path):
        candidates = sorted(glob.glob(os.path.join(path, "ckpt_step*.pt")))
        if candidates:
            return max(candidates, key=lambda p: int(os.path.basename(p).split("step")[-1].split(".")[0]))
        best = os.path.join(path, "ckpt_best.pt")
        if os.path.exists(best):
            return best
    raise FileNotFoundError(f"Checkpoint not found: {path}")


def _infer_cfg_from_state(state, default_vocab_size: int) -> RyuuGPTConfig:
    emb = state.get("token_emb.weight")
    if emb is not None and hasattr(emb, "shape") and len(emb.shape) == 2:
        vocab_size, n_embd = int(emb.shape[0]), int(emb.shape[1])
    else:
        vocab_size, n_embd = default_vocab_size, 768

    block_keys = [k for k in state.keys() if k.startswith("blocks.")]
    block_ids = []
    for key in block_keys:
        parts = key.split(".")
        if len(parts) > 1 and parts[1].isdigit():
            block_ids.append(int(parts[1]))
    n_layer = (max(block_ids) + 1) if block_ids else 16

    mask_key = next((k for k in state.keys() if k.endswith("attn.mask")), None)
    context_size = int(state[mask_key].shape[-1]) if mask_key is not None else 1024

    head_dim = 64
    rotary_key = next((k for k in state.keys() if k.endswith("attn.rotary.cos")), None)
    if rotary_key is not None and len(state[rotary_key].shape) >= 4:
        head_dim = int(state[rotary_key].shape[-1])
    n_head = max(1, n_embd // max(1, head_dim))

    return RyuuGPTConfig(
        vocab_size=vocab_size,
        context_size=context_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        use_reasoning_head=any(k.startswith("reasoning_head.") for k in state.keys()),
        use_value_head=any(k.startswith("value_head.") for k in state.keys()),
    )

class RyuuDevReasoner:
    def __init__(
        self,
        model_ckpt: str,
        tokenizer_path: str,
        device: Optional[str] = None,
        session_max_turns: int = 16,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"🧠 Initializing RyuuDev Reasoner (with memory) on {self.device}")

        # tokenizer & model
        self.tokenizer = BPETokenizer.load(tokenizer_path)
        ckpt_path = _resolve_checkpoint(model_ckpt)
        state = torch.load(ckpt_path, map_location=self.device)
        state_dict = _extract_state_dict(state)

        cfg_dict = {}
        if isinstance(state, dict):
            cfg_dict = state.get("config") or state.get("model_config") or {}
        cfg = RyuuGPTConfig(**cfg_dict) if cfg_dict else _infer_cfg_from_state(state_dict, self.tokenizer.vocab_size)

        self.model = RyuuGPT(cfg).to(self.device)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

        # memory pieces
        self.session = SessionMemory(max_turns=session_max_turns, max_chars=16_000)
        self.store = SnippetStore()
        self.ctx = ContextBuilder(self.session, self.store, max_ctx_chars=4000, max_snippets=6)

        logging.info("✅ Reasoner ready (memory + retrieval enabled)")

    # ---------- intent ----------
    def classify(self, prompt: str):
        p = prompt.lower()
        if "debug" in p or "error" in p or ERROR_HINT.search(prompt):
            return "debug"
        if "explain" in p or p.strip().startswith("why"):
            return "explain"
        if "write" in p or "generate" in p or "function" in p or CODE_FENCE.search(prompt):
            return "generate"
        return "general"

    def _boost(self, intent: str) -> str:
        return {
            "debug":    "Think step-by-step. Identify the bug, explain cause, then output a fixed code block.",
            "explain":  "Explain like you're mentoring a junior dev. Use headings, short steps, and examples.",
            "generate": "Produce clean, unique, runnable code with a brief docstring and tests if useful.",
            "general":  "Be clear, concise, and technically accurate.",
        }[intent]

    # ---------- run ----------
    @torch.no_grad()
    def run(self, user_prompt: str, max_tokens=256, temperature=0.7, top_k=50):
        intent = self.classify(user_prompt)

        # build context from memory & snippets
        context_block = self.ctx.build(user_prompt)
        sys_preamble = (
            f"{self._boost(intent)}\n"
            f"Use the context to answer. If context is irrelevant, ignore it.\n"
            f"---\n{context_block}\n---\n"
        )
        enhanced = f"{sys_preamble}\nUser Query:\n{user_prompt}\nAssistant:"

        # encode/generate
        ids = torch.tensor([self.tokenizer.encode_ids(enhanced)], dtype=torch.long, device=self.device)
        out = self.model.generate(ids, max_new_tokens=max_tokens, temperature=temperature, top_k=top_k)
        text = self.tokenizer.decode(out[0].tolist())
        answer = self._postprocess(text, enhanced)

        # update memory (turns + snippet extraction)
        self.session.add("user", user_prompt)
        self.session.add("assistant", answer)
        self._harvest_snippets(user_prompt, answer)

        return {
            "intent": intent,
            "context_used": context_block,
            "output": answer.strip(),
        }

    # ---------- helpers ----------
    def _postprocess(self, full_text: str, prompt_prefix: str) -> str:
        # cut the echoed prompt if present
        if full_text.startswith(prompt_prefix):
            full_text = full_text[len(prompt_prefix):]
        # clean artifacts
        cleaned = re.sub(r"<\|endoftext\|>", "", full_text)
        cleaned = re.sub(r"```+", "```", cleaned).strip()
        return cleaned

    def _harvest_snippets(self, user: str, assistant: str):
        # store code blocks from either side
        for m in CODE_FENCE.finditer(user):
            self.ctx.add_code(m.group(0), source="user")
        for m in CODE_FENCE.finditer(assistant):
            self.ctx.add_code(m.group(0), source="assistant")
        # store error-like text
        if ERROR_HINT.search(user):
            self.ctx.add_error(user, source="user", where="prompt")
        if ERROR_HINT.search(assistant):
            self.ctx.add_error(assistant, source="assistant", where="answer")
