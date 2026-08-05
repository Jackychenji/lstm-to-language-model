"""Decompose the LSTM-vs-Transformer wall-clock gap into architecture and
implementation.

The trained LSTM unrolls its recurrence in a Python loop; the Transformer calls
a fused attention kernel. Comparing only those two conflates "attention
parallelises over time" with "one of these runs a fused CUDA kernel and the
other runs 128 Python iterations". Adding cuDNN's fused nn.LSTM as a third
reference separates the two:

    hand-written LSTM  vs  nn.LSTM        -> implementation cost
    nn.LSTM            vs  Transformer    -> architecture, both fused

nn.LSTM is used here only as a timing reference. The trained model in
lstm-lm/ does not use it.

    python benchmark_speed.py
"""
from __future__ import annotations

import argparse
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, "lstm-lm")
from model import CharLSTMLM  # noqa: E402



def load_transformer_class():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "tf_model", "transformer-lm/model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CharTransformerLM


class FusedLSTMLM(nn.Module):
    """Same shape as CharLSTMLM but with cuDNN's fused nn.LSTM.

    Timing reference only -- not the trained model. Parameter count differs
    slightly because nn.LSTM carries two bias vectors per gate group where the
    hand-written layer carries one.
    """

    def __init__(self, vocab_size, emb_dim=256, hidden_dim=512, num_layers=2,
                 dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, num_layers=num_layers,
                            dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        h, _ = self.lstm(self.drop(self.embedding(x)))
        return self.head(self.drop(h))

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


def time_step(model, x, y, device, steps, warmup, unwrap, use_amp):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    def one():
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device, dtype=torch.bfloat16,
                            enabled=use_amp):
            out = model(x)
            logits = out[0] if unwrap else out
            loss = criterion(logits.reshape(-1, logits.size(-1)).float(),
                             y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    for _ in range(warmup):
        one()
    if device == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(steps):
        one()
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / steps


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    use_amp = device == "cuda"

    CharTransformerLM = load_transformer_class()

    x = torch.randint(0, args.vocab, (args.batch, args.seq_len), device=device)
    y = torch.randint(0, args.vocab, (args.batch, args.seq_len), device=device)

    models = [
        ("LSTM, hand-written loop (trained)",
         CharLSTMLM(args.vocab).to(device), True),
        ("LSTM, cuDNN fused nn.LSTM (reference)",
         FusedLSTMLM(args.vocab).to(device), False),
        ("Transformer, fused SDPA (trained)",
         CharTransformerLM(args.vocab, max_len=args.seq_len).to(device), False),
    ]

    print(f"device: {torch.cuda.get_device_name(0) if device == 'cuda' else 'cpu'}")
    print(f"batch {args.batch} x seq {args.seq_len}, bf16 autocast={use_amp}, "
          f"{args.steps} timed steps after {args.warmup} warmup\n")
    print(f"{'model':<40} {'params':>10} {'ms/step':>9} {'rel':>7}")
    print("-" * 70)

    results = []
    for name, model, unwrap in models:
        ms = 1000 * time_step(model, x, y, device, args.steps, args.warmup,
                              unwrap, use_amp)
        results.append((name, model.num_parameters(), ms))
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    base = results[0][2]
    for name, params, ms in results:
        print(f"{name:<40} {params:>10,} {ms:>9.1f} {base / ms:>6.2f}x")

    hand, fused, transformer = (r[2] for r in results)
    print(f"\nimplementation (hand-written loop -> fused nn.LSTM): "
          f"{hand / fused:.2f}x")
    print(f"architecture   (fused nn.LSTM -> Transformer):        "
          f"{fused / transformer:.2f}x")
    print(f"combined       (hand-written loop -> Transformer):    "
          f"{hand / transformer:.2f}x")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab", type=int, default=4955)
    parser.add_argument("--batch", type=int, default=96)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=10)
    main(parser.parse_args())
