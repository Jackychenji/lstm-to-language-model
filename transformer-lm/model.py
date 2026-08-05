"""Character-level causal Transformer language model.

Decoder-only, built to match the LSTM in ../lstm-lm at comparable parameter
count so the two architectures can be compared on the same corpus, vocabulary,
token budget and evaluation. Blocks are written out rather than using
nn.TransformerEncoder; the attention kernel itself is
F.scaled_dot_product_attention with is_causal=True.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


class Block(nn.Module):
    """Pre-LayerNorm block: x + attn(ln(x)), then x + ffn(ln(x))."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class CharTransformerLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 256, n_layers: int = 6,
                 n_heads: int = 8, d_ff: int = 1024, max_len: int = 128,
                 dropout: float = 0.2):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            Block(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)  # untied

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        """x: (B, T) int64 -> logits (B, T, vocab_size)"""
        B, T = x.size()
        if T > self.max_len:
            raise ValueError(f"sequence length {T} exceeds max_len {self.max_len}")

        pos = torch.arange(T, device=x.device)
        h = self.drop(self.token_emb(x) + self.pos_emb(pos))
        for block in self.blocks:
            h = block(h)
        return self.head(self.ln_f(h))

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def generate(self, prompt_ids, max_new_tokens: int, temperature: float = 1.0,
                 top_k: int | None = None, device: str = "cpu"):
        self.eval()
        ids = torch.tensor(prompt_ids, dtype=torch.long,
                           device=device).unsqueeze(0)

        for _ in range(max_new_tokens):
            # no KV cache; recompute over the trailing max_len characters
            window = ids[:, -self.max_len:]
            logits = self(window)[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None and 0 < top_k < logits.size(-1):
                kth = torch.topk(logits, top_k, dim=-1).values[:, -1:]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            probs = torch.softmax(logits, dim=-1)
            ids = torch.cat([ids, torch.multinomial(probs, num_samples=1)], dim=1)

        return ids[0].tolist()
