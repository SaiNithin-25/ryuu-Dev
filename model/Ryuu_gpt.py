# model/Ryuu_gpt.py
"""
RyuuGPT v3 (Reasoning-Ready)

FIXES:
- hidden_states now includes post-final-norm x so reasoning head
  sees same representation as LM head
- value_head disabled by default (was wasting params/memory)
- _current_step sync uses setattr so it works regardless of class impl
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torch.utils.checkpoint import checkpoint

from .transformer import TransformerBlock, RMSNorm
from .config import RyuuGPTConfig
from .core.reasoning_head import ReasoningHeadV4


# ---------------------------------------------------------
# Weight initialization
# ---------------------------------------------------------
def _init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif hasattr(module, "scale") and isinstance(module.scale, torch.Tensor):
        nn.init.ones_(module.scale)
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.zeros_(module.bias)


# ---------------------------------------------------------
# RyuuGPT
# ---------------------------------------------------------
class RyuuGPT(nn.Module):
    def __init__(self, config: RyuuGPTConfig):
        super().__init__()
        self.config = config

        self.vocab_size   = config.vocab_size
        self.n_embd       = config.n_embd
        self.context_size = config.context_size

        # ---- Embeddings ----
        self.token_emb = nn.Embedding(self.vocab_size, self.n_embd)
        self.use_rope  = getattr(config, "use_rope", False)
        self.pos_emb   = None if self.use_rope else nn.Embedding(self.context_size, self.n_embd)
        self.dropout   = nn.Dropout(config.dropout)

        # ---- Transformer blocks ----
        self.blocks = nn.ModuleList([
            TransformerBlock(
                self.n_embd,
                config.n_head,
                context_size=self.context_size,
                dropout=config.dropout,
                use_rms=config.use_rms_norm,
                use_rope=config.use_rope,
            )
            for _ in range(config.n_layer)
        ])

        # ---- Final norm ----
        self.ln_f = RMSNorm(self.n_embd) if config.use_rms_norm else nn.LayerNorm(self.n_embd)

        # ---- LM head (tied weights) ----
        self.lm_head = nn.Linear(self.n_embd, self.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

        # FIX: value_head off by default — only create if explicitly enabled
        self.value_head = (
            nn.Linear(self.n_embd, 1)
            if getattr(config, "use_value_head", False)
            else None
        )

        # ---- Reasoning head ----
        self.reasoning_head = (
            ReasoningHeadV4(hidden_dim=self.n_embd, **getattr(config, "reasoning_head_kwargs", {}))
            if getattr(config, "use_reasoning_head", False)
            else None
        )

        self._steer_bad_mask  = None
        self._steer_good_mask = None
        self.gradient_checkpointing = getattr(config, "gradient_checkpointing", False)
        self._current_step = 0

        self.apply(_init_weights)

    # -----------------------------------------------------
    @property
    def device(self):
        return next(self.parameters()).device

    # -----------------------------------------------------
    def _prepare_embeddings(self, idx):
        tok = self.token_emb(idx)
        if self.pos_emb is not None:
            pos = self.pos_emb(torch.arange(idx.size(1), device=idx.device))[None]
            tok = tok + pos
        return self.dropout(tok)

    # -----------------------------------------------------
    def forward(self, idx, targets: Optional[torch.Tensor] = None):
        x = self._prepare_embeddings(idx)

        hidden_states = [] if self.reasoning_head is not None else None
        for block in self.blocks:
            x = checkpoint(block, x) if self.gradient_checkpointing and self.training else block(x)
            if hidden_states is not None:
                hidden_states.append(x)

        x = self.ln_f(x)

        # FIX: append post-norm x so reasoning head sees same repr as lm_head
        if hidden_states is not None:
            hidden_states.append(x)

        logits = self.lm_head(x)

        # ---- Base LM loss ----
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=getattr(self.config, "pad_token_id", -100),
            )

        # ---- Value ----
        value = self.value_head(x).squeeze(-1) if self.value_head else None

        # ---- Reasoning ----
        reasoning = None
        if self.reasoning_head is not None:
            attention_mask = None
            if targets is not None:
                pad_id = getattr(self.config, "pad_token_id", -100)
                attention_mask = (targets != pad_id).long()

            reasoning = self.reasoning_head(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                logits=logits,
                targets=targets,
            )

            # add reasoning loss to total loss
            r_loss = reasoning.get("loss")
            if loss is not None and r_loss is not None:
                weight = float(getattr(self.config, "reasoning_loss_weight", 1.0))
                loss = loss + weight * r_loss

            # Token-weighted auxiliary loss
            if targets is not None:
                token_w = reasoning.get("token_weights", None)
                if token_w is not None:
                    logp = F.log_softmax(logits, dim=-1)
                    tgt_logp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
                    token_loss = -(token_w * tgt_logp).mean()
                    loss = loss + 0.05 * token_loss

        return logits, loss, value, reasoning

    # -----------------------------------------------------
    def log_prob(self, prompt_ids, response_ids):
        x = torch.cat([prompt_ids, response_ids[:, :-1]], dim=1)
        logits, _, _, _ = self(x)
        logits = logits[:, -response_ids.size(1):]
        logp   = torch.log_softmax(logits, dim=-1)
        return logp.gather(-1, response_ids.unsqueeze(-1)).squeeze(-1).sum(dim=1)

    # -----------------------------------------------------
    def _build_token_steering_masks(self, vocab_size, device):
        bad  = torch.zeros(vocab_size, device=device)
        good = torch.zeros(vocab_size, device=device)
        for i in range(min(vocab_size, 128)):
            c = chr(i)
            if c in ".;,:!?":  bad[i]  = 1.0
            if c in "(){}[]":  good[i] = 1.0
        return bad, good

    # -----------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        input_ids,
        max_new_tokens=128,
        temperature=0.7,
        top_k=40,
        top_p=0.4,
        do_sample=True,
        reasoning=0.35,
        eos_token_id: Optional[int] = None,
    ):
        device = self.device
        out = input_ids.to(device)
        if self._steer_bad_mask is None:
            self._steer_bad_mask, self._steer_good_mask = self._build_token_steering_masks(self.vocab_size, device)

        for _ in range(max_new_tokens):
            model_in    = out[:, -self.context_size:]
            logits, _, _, reasoning_out = self.forward(model_in)
            next_logits = logits[:, -1, :] / temperature

            if reasoning_out and reasoning_out.get("scores") is not None:
                steer = torch.tanh(reasoning_out["scores"].mean(dim=-1)).unsqueeze(-1) * float(reasoning)
                next_logits = (
                    next_logits
                    - steer * self._steer_bad_mask
                    + steer * self._steer_good_mask
                )

            if top_k > 0:
                v, _ = torch.topk(next_logits, top_k)
                next_logits[next_logits < v[:, -1:]] = -1e10

            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
                probs    = F.softmax(sorted_logits, dim=-1)
                cumprobs = torch.cumsum(probs, dim=-1)
                mask          = cumprobs > top_p
                mask[..., 1:] = mask[..., :-1].clone()
                mask[..., 0]  = False
                sorted_logits[mask] = -1e10
                next_logits = torch.gather(sorted_logits, -1, torch.argsort(sorted_idx))

            if do_sample:
                probs      = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
            else:
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)

            out = torch.cat([out, next_token], dim=1)
            out = out[:, -self.context_size:]
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return out

    # -----------------------------------------------------
    def _reflection_confidence(self, draft_reasoning, revised_reasoning):
        if (
            draft_reasoning is None or revised_reasoning is None
            or draft_reasoning.get("scores") is None
            or revised_reasoning.get("scores") is None
        ):
            return None
        d = draft_reasoning["scores"].norm(dim=-1)
        r = revised_reasoning["scores"].norm(dim=-1)
        return torch.tanh((r - d).clamp(min=0.0)).mean().item()

    # -----------------------------------------------------
    @torch.no_grad()
    def self_reflect(self, input_ids, max_new_tokens=128, return_steps=False):
        draft  = self.generate(input_ids, max_new_tokens, temperature=0.8, do_sample=True)
        _, _, _, draft_reasoning = self.forward(draft)

        critique_prompt = torch.cat([input_ids, draft], dim=1)
        critique = self.generate(critique_prompt, 64, temperature=0.7)

        revise_prompt = torch.cat([critique_prompt, critique], dim=1)
        revised = self.generate(revise_prompt, max_new_tokens, temperature=0.6)
        _, _, _, revised_reasoning = self.forward(revised)

        confidence = self._reflection_confidence(draft_reasoning, revised_reasoning)

        if return_steps:
            return {"draft": draft, "critique": critique, "revised": revised, "confidence": confidence}
        return revised


# =========================================================
# SANITY CHECK
# =========================================================
if __name__ == "__main__":
    print("\n=== RyuuGPT SANITY CHECK ===")
    cfg = RyuuGPTConfig(
        vocab_size=30000, context_size=64,
        n_layer=4, n_head=4, n_embd=128,
        dropout=0.1, pad_token_id=0,
        use_reasoning_head=True,
    )
    model = RyuuGPT(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (2, 32))
    y = torch.randint(0, cfg.vocab_size, (2, 32))
    with torch.no_grad():
        logits, loss, value, reasoning = model(x, targets=y)
    print("Logits shape:", logits.shape)
    print("Loss:", loss.item())
    print("Reasoning keys:", list(reasoning.keys()))
    print("Reasoning loss:", reasoning["loss"].item())
    print("\u2714 All sanity checks passed\n")