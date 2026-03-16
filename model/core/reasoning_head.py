# model/core/reasoning_head.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReasoningHeadV4(nn.Module):
    """
    ReasoningHead V4.1 — Curriculum + Token-level Signals

    FIXES:
    - _current_step now stored as plain int attribute (was silently ignored)
    - scale==0.0 gate now returns loss=torch.tensor(0.0) instead of None
      so train_reason_ema actually accumulates from step 0
    - confidence_head indentation fixed (was broken __init__ layout)
    - reg_head loss now uses .abs().mean() instead of .pow(2).mean()
      to avoid squaring already-small values into near-zero
    """

    def __init__(
        self,
        hidden_dim: int,
        reasoning_dim: int = 16,
        num_layers_used: int = 4,
        dropout: float = 0.1,
        warmup_steps: int = 3_000,
        full_steps: int = 12_000,
        token_signal_dim: int = 3,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.reasoning_dim = reasoning_dim
        self.num_layers_used = num_layers_used
        self.warmup_steps = warmup_steps
        self.full_steps = full_steps

        # FIX: store as plain int so trainer assignment works
        self._current_step = 0

        # ---- Layer importance ----
        self.layer_logits = nn.Parameter(torch.zeros(num_layers_used))

        # ---- Token-signal encoder ----
        self.token_signal_proj = nn.Sequential(
            nn.Linear(token_signal_dim, hidden_dim),
            nn.GELU(),
        )

        # ---- Confidence head (indentation fixed) ----
        self.confidence_head = nn.Sequential(
            nn.Linear(reasoning_dim, reasoning_dim // 2),
            nn.GELU(),
            nn.Linear(reasoning_dim // 2, 1),
            nn.Sigmoid(),
        )

        # ---- Token-level attention ----
        self.token_attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # ---- Projection to reasoning space ----
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, reasoning_dim),
        )

        # ---- Auxiliary regularizer ----
        self.reg_head = nn.Linear(reasoning_dim, 1)

    # --------------------------------------------------
    def _curriculum_scale(self, step: int):
        if step < self.warmup_steps:
            return 0.0, 0
        elif step < self.full_steps:
            s = (step - self.warmup_steps) / max(1, self.full_steps - self.warmup_steps)
            return float(s), 1
        else:
            return 1.0, 2

    # --------------------------------------------------
    def _compute_token_signals(self, logits, targets):
        with torch.no_grad():
            log_probs = F.log_softmax(logits, dim=-1)
            probs = log_probs.exp()
            nll = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            entropy = -(probs * log_probs).sum(dim=-1)
            top2 = torch.topk(probs, k=2, dim=-1).values
            conf_gap = top2[..., 0] - top2[..., 1]
        return torch.stack([nll, entropy, conf_gap], dim=-1)

    # --------------------------------------------------
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        logits=None,
        targets=None,
    ):
        layers = hidden_states[-self.num_layers_used:]
        B, T, D = layers[0].shape
        device = layers[0].device

        step = int(self._current_step) if self._current_step is not None else 0
        scale, phase = self._curriculum_scale(step)
        layer_weights = F.softmax(self.layer_logits, dim=0)

        # FIX: return loss=zero tensor (not None) so EMA accumulates
        if scale == 0.0:
            return {
                "scores": torch.zeros(B, self.reasoning_dim, device=device),
                "loss": torch.tensor(0.0, device=device),
                "info": {
                    "phase": 0,
                    "scale": 0.0,
                    "layer_weights": layer_weights.detach(),
                },
            }

        # ---- Layer fusion ----
        top_bias = torch.zeros_like(layer_weights)
        top_bias[-1] = 1.0
        layer_weights = (1 - scale) * top_bias + scale * layer_weights
        fused = sum(w * h for w, h in zip(layer_weights, layers))

        # ---- Token-level signals ----
        if logits is not None and targets is not None:
            token_signals = self._compute_token_signals(logits, targets)
            token_signal_emb = self.token_signal_proj(token_signals)
            fused = fused + scale * token_signal_emb
        else:
            token_signals = None

        # ---- Token attention ----
        token_scores = self.token_attn(fused).squeeze(-1)
        if attention_mask is not None:
            token_scores = token_scores.masked_fill(attention_mask == 0, -1e9)

        temperature = max(0.5, 1.0 - scale)
        token_weights = F.softmax(token_scores / temperature, dim=-1)
        pooled = torch.sum(fused * token_weights.unsqueeze(-1), dim=1)

        # ---- Reasoning vector ----
        reasoning_vec = self.proj(pooled)
        confidence = self.confidence_head(reasoning_vec).squeeze(-1)

        # FIX: use .abs().mean() — pow(2) collapses tiny values to near-zero
        reg_logits = self.reg_head(reasoning_vec).squeeze(-1)
        base_loss = reg_logits.abs().mean()
        reasoning_loss = scale * base_loss

        # ---- Diagnostics ----
        info = {
            "phase": phase,
            "scale": scale,
            "layer_weights": layer_weights.detach(),
            "token_entropy": (
                -(token_weights * torch.log(token_weights + 1e-8)).sum(-1).mean()
            ).detach(),
        }

        if token_signals is not None:
            info["avg_nll"]      = token_signals[..., 0].mean().detach()
            info["avg_entropy"]  = token_signals[..., 1].mean().detach()
            info["avg_conf_gap"] = token_signals[..., 2].mean().detach()

        return {
            "scores":       reasoning_vec,
            "loss":         reasoning_loss,
            "confidence":   confidence.mean().detach(),
            "info":         info,
            "token_weights": token_weights.detach(),
        }