"""Check that the Transformer's causal mask actually holds.

Perturbing the token at position t must leave every logit at positions < t
bit-identical, and must change at least one logit at positions >= t. Run this
after any change to the attention path.

    python test_causality.py
"""
from __future__ import annotations

import argparse
import sys

import torch

from model import CharTransformerLM


def check(model, x, position, device):
    model.eval()
    with torch.no_grad():
        base = model(x)
        perturbed = x.clone()
        perturbed[:, position] = (perturbed[:, position] + 7) % model.vocab_size
        other = model(perturbed)

    before = (base[:, :position] - other[:, :position]).abs().max().item() \
        if position > 0 else 0.0
    after = (base[:, position:] - other[:, position:]).abs().max().item()
    return before, after


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    model = CharTransformerLM(args.vocab, max_len=args.seq_len).to(device)
    x = torch.randint(0, args.vocab, (args.batch, args.seq_len), device=device)

    print(f"device: {device}   batch {args.batch} x seq {args.seq_len}\n")
    print(f"{'perturbed position':>19} | {'max |d| before':>15} | "
          f"{'max |d| at/after':>16} | verdict")
    print("-" * 72)

    failures = 0
    positions = [0, 1, args.seq_len // 4, args.seq_len // 2,
                 args.seq_len - 2, args.seq_len - 1]
    for p in positions:
        before, after = check(model, x, p, device)
        leaked = before != 0.0
        inert = after == 0.0 and p < args.seq_len
        ok = not leaked and not inert
        failures += not ok
        verdict = "ok" if ok else ("LEAK" if leaked else "no effect at/after")
        print(f"{p:>19} | {before:>15.3e} | {after:>16.3e} | {verdict}")

    print()
    if failures:
        print(f"FAILED: {failures} position(s) violate causality")
        sys.exit(1)
    print("PASSED: information never flows backwards; "
          "each position influences itself and later positions only")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab", type=int, default=4955)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=128)
    main(parser.parse_args())
