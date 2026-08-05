"""Decompose the LSTM-vs-Transformer wall-clock gap into architecture and
implementation.

The trained LSTM unrolls its recurrence in a Python loop; the Transformer calls
a fused attention kernel. Comparing only those two conflates "attention
parallelises over time" with "one of these runs a fused CUDA kernel and the
other runs 128 Python iterations". Adding cuDNN's fused nn.LSTM as a third
reference separates the dominant effect:

    hand-written LSTM  vs  nn.LSTM        -> cost of the Python recurrence
    nn.LSTM            vs  Transformer    -> two fused implementations

Note what the second row is and is not. Both sides are optimised kernels, but
they are *different* kernels with different operator mixes and different
amounts of vendor tuning. The ratio is a comparison of two fused
implementations, not a clean measurement of architecture with everything else
held constant.

nn.LSTM is used here only as a timing reference. The trained model in
lstm-lm/ does not use it.

    python benchmark_speed.py --repeats 3
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

    # Rounds are interleaved (A,B,C, A,B,C, ...) rather than grouped
    # (A,A,A, B,B,B, ...). On a laptop GPU sustained load causes thermal
    # throttling, so grouping would charge whichever model ran first for
    # everyone else's heat. Ratios are taken within a round, where all three
    # models saw the same clocks.
    per_round = []
    for r in range(args.repeats):
        row = [time_step(model, x, y, device, args.steps, args.warmup,
                         unwrap, use_amp) * 1000
               for _, model, unwrap in models]
        per_round.append(row)

    def median(xs):
        s = sorted(xs)
        return s[len(s) // 2] if len(s) % 2 else (s[len(s) // 2 - 1]
                                                  + s[len(s) // 2]) / 2

    times = list(zip(*per_round))
    base = median(times[0])
    for (name, model, _), t in zip(models, times):
        spread = f" ±{(max(t) - min(t)) / 2:>5.1f}" if len(t) > 1 else ""
        print(f"{name:<40} {model.num_parameters():>10,} "
              f"{median(t):>9.1f}{spread} {base / median(t):>6.2f}x")

    def within_round(i, j):
        rs = [row[i] / row[j] for row in per_round]
        return median(rs), min(rs), max(rs)

    hand_fused = within_round(0, 1)
    fused_fused = within_round(1, 2)
    combined = within_round(0, 2)

    print(f"\nratios taken within each round, median (range) over "
          f"{args.repeats} rounds")
    print(f"  cost of the Python recurrence (hand-written -> nn.LSTM):    "
          f"{hand_fused[0]:.2f}x ({hand_fused[1]:.2f}-{hand_fused[2]:.2f})")
    print(f"  fused vs fused                (nn.LSTM -> Transformer):     "
          f"{fused_fused[0]:.2f}x ({fused_fused[1]:.2f}-{fused_fused[2]:.2f})")
    print(f"  combined                      (hand-written -> Transformer):"
          f"{combined[0]:.2f}x ({combined[1]:.2f}-{combined[2]:.2f})")
    print("\nThe second row compares two different optimised kernels, not "
          "architecture with\neverything else held constant.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab", type=int, default=4955)
    parser.add_argument("--batch", type=int, default=96)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    main(parser.parse_args())
