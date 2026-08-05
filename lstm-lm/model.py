"""Character-level LSTM language model, built on the hand-written LSTM from
CS324 Assignment 3 Part 1. torch.nn.LSTM is not used."""
from __future__ import annotations

import torch
import torch.nn as nn


class LSTMLayer(nn.Module):
    """One LSTM layer, unrolled over time.

    Gate order along the last axis is g, i, f, o. The assignment used eight
    separate Linear layers; fusing them into two is equivalent and ~4x faster.
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.Wx = nn.Linear(input_dim, 4 * hidden_dim)
        self.Wh = nn.Linear(hidden_dim, 4 * hidden_dim, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        std = 1.0 / (self.hidden_dim ** 0.5)
        for weight in (self.Wx.weight, self.Wh.weight):
            nn.init.uniform_(weight, -std, std)
        nn.init.zeros_(self.Wx.bias)
        # forget gate bias = 1, so the cell state survives early training
        with torch.no_grad():
            self.Wx.bias[2 * self.hidden_dim:3 * self.hidden_dim].fill_(1.0)

    def forward(self, x, state=None):
        batch_size, seq_length, _ = x.size()
        if state is None:
            h_t = x.new_zeros(batch_size, self.hidden_dim)
            c_t = x.new_zeros(batch_size, self.hidden_dim)
        else:
            h_t, c_t = state

        xs = self.Wx(x)  # whole sequence at once; doesn't depend on h

        outputs = []
        for t in range(seq_length):
            gates = xs[:, t] + self.Wh(h_t)
            g_t, i_t, f_t, o_t = gates.chunk(4, dim=-1)

            g_t = torch.tanh(g_t)
            i_t = torch.sigmoid(i_t)
            f_t = torch.sigmoid(f_t)
            o_t = torch.sigmoid(o_t)

            c_t = g_t * i_t + c_t * f_t
            h_t = torch.tanh(c_t) * o_t
            outputs.append(h_t)

        return torch.stack(outputs, dim=1), (h_t, c_t)


class CharLSTMLM(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 256,
                 hidden_dim: int = 512, num_layers: int = 2,
                 dropout: float = 0.2):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.layers = nn.ModuleList([
            LSTMLayer(emb_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, vocab_size)

        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x, state=None):
        """x: (B, T) int64 -> logits (B, T, vocab_size), state"""
        h = self.dropout(self.embedding(x))

        new_state = []
        for i, layer in enumerate(self.layers):
            h, layer_state = layer(h, None if state is None else state[i])
            h = self.dropout(h)
            new_state.append(layer_state)

        return self.head(h), new_state

    @staticmethod
    def detach_state(state):
        if state is None:
            return None
        return [(h.detach(), c.detach()) for h, c in state]

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def generate(self, prompt_ids, max_new_tokens: int, temperature: float = 1.0,
                 top_k: int | None = None, device: str = "cpu"):
        self.eval()
        ids = torch.tensor(prompt_ids, dtype=torch.long,
                           device=device).unsqueeze(0)

        # consume the prompt once, then advance one character per step
        logits, state = self(ids)
        out = list(prompt_ids)

        for _ in range(max_new_tokens):
            step_logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None and 0 < top_k < step_logits.size(-1):
                kth = torch.topk(step_logits, top_k, dim=-1).values[:, -1:]
                step_logits = step_logits.masked_fill(step_logits < kth,
                                                      float("-inf"))
            probs = torch.softmax(step_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            out.append(int(next_id))
            logits, state = self(next_id, state)

        return out
