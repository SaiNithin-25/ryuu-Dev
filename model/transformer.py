# model/transformer.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------
# Utility: RMSNorm
# --------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.pow(2).mean(-1, keepdim=True).sqrt()
        return x / (norm + self.eps) * self.scale


# --------------------------
# Rotary Embeddings (RoPE)
# --------------------------
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position=2048, base=10000):
        super().__init__()
        assert dim % 2 == 0, "RoPE dimension must be even"
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_position, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos", emb.cos()[None, None, :, :])
        self.register_buffer("sin", emb.sin()[None, None, :, :])

    def rotate(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.size(2)
        cos = self.cos[:, :, :seq_len, :].to(x.device)
        sin = self.sin[:, :, :seq_len, :].to(x.device)
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        xr = torch.stack((-x2, x1), dim=-1).flatten(-2)
        return x * cos + xr * sin


# --------------------------
# Multi-Head Attention
# --------------------------
class SelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, context_size, dropout=0.1, use_rope=False):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_size, context_size), 1).bool())
        self.use_rope = use_rope
        self.rotary = RotaryEmbedding(self.head_dim, max_position=context_size) if use_rope else None
        self._flash = hasattr(F, "scaled_dot_product_attention")

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_head, self.head_dim).transpose(1, 3)
        q, k, v = qkv.unbind(2)  # (B, nh, T, hs)

        if self.use_rope:
            q = self.rotary.rotate(q, T)
            k = self.rotary.rotate(k, T)

        if self._flash:
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=self.mask[:T, :T].to(q.device),
                dropout_p=self.dropout.p if self.training else 0.0
            )
        else:
            att = (q @ k.transpose(-2, -1)) * self.scale
            att = att.masked_fill(self.mask[:T, :T].to(att.device), float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


# --------------------------
# Feed Forward (SwiGLU)
# --------------------------
class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout=0.1):
        super().__init__()
        hidden = 4 * n_embd
        self.fc1 = nn.Linear(n_embd, hidden)
        self.fc2 = nn.Linear(hidden // 2, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x1, x2 = self.fc1(x).chunk(2, dim=-1)
        return self.dropout(self.fc2(F.silu(x1) * x2))


# --------------------------
# Transformer Block
# --------------------------
class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head, context_size, dropout=0.1, use_rms=True, use_rope=True):
        super().__init__()
        Norm = RMSNorm if use_rms else nn.LayerNorm
        self.norm1 = Norm(n_embd)
        self.norm2 = Norm(n_embd)
        self.attn = SelfAttention(n_embd, n_head, context_size, dropout, use_rope)
        self.ff = FeedForward(n_embd, dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


# --------------------------
# Full Model: RyuuTransformer
# --------------------------
class RyuuTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        n_embd=768,
        n_layer=16,
        n_head=12,
        context_size=1024,
        dropout=0.1,
        use_rms=True,
        use_rope=True,
        use_value_head=False
    ):
        super().__init__()
        self.context_size = context_size
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(context_size, n_embd)
        self.blocks = nn.ModuleList([
            TransformerBlock(n_embd, n_head, context_size, dropout, use_rms, use_rope)
            for _ in range(n_layer)
        ])
        self.ln_f = RMSNorm(n_embd) if use_rms else nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        self.head.weight = self.token_emb.weight  # weight tying
        self.value_head = nn.Linear(n_embd, 1) if use_value_head else None
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None, return_value=False):
        B, T = idx.shape
        if T > self.context_size:
            idx = idx[:, -self.context_size:]
            T = self.context_size

        tok = self.token_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        x = tok + pos

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        value = None
        if return_value and self.value_head is not None:
            value = self.value_head(x).squeeze(-1)

        return logits, loss, value
