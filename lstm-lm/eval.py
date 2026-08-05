"""Score the model against unigram baselines, against a shuffled copy of the
held-out text, and in a controlled context-length ablation.

    python eval.py --ckpt runs/best.pt
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter

import torch
import torch.nn.functional as F

from data import build_datasets
from sample import load
from utils import bits_per_char, perplexity

LN2 = math.log(2.0)


@torch.no_grad()
def corpus_nll(model, ids, device, seq_len=256, batch_size=32, max_windows=None):
    """Mean next-char cross-entropy over a flat id array."""
    windows = []
    for start in range(0, len(ids) - seq_len - 1, seq_len):
        windows.append(ids[start:start + seq_len + 1])
        if max_windows and len(windows) >= max_windows:
            break
    if not windows:
        raise ValueError("text too short to evaluate")

    total, count = 0.0, 0
    for i in range(0, len(windows), batch_size):
        chunk = torch.tensor(
            [w.tolist() for w in windows[i:i + batch_size]],
            dtype=torch.long, device=device)
        logits, _ = model(chunk[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               chunk[:, 1:].reshape(-1), reduction="sum")
        total += loss.item()
        count += chunk[:, 1:].numel()
    return total / count


def unigram_nll(train_ids, val_ids, vocab_size, alpha=1.0):
    """Cross-entropy on val of a unigram model fitted on train, add-alpha
    smoothed. This is the baseline a real unigram model would achieve."""
    counts = Counter(int(i) for i in train_ids)
    denom = len(train_ids) + alpha * vocab_size
    logp = [math.log((counts.get(t, 0) + alpha) / denom) for t in range(vocab_size)]
    return -sum(logp[int(i)] for i in val_ids) / len(val_ids)


def unigram_entropy(ids):
    """Entropy of the val distribution itself -- the lowest cross-entropy any
    unigram model could reach on this text, so a strictly harder baseline."""
    counts = Counter(int(i) for i in ids)
    n = sum(counts.values())
    return -sum((c / n) * math.log(c / n) for c in counts.values())


@torch.no_grad()
def context_ablation(model, ids, device, short=128, long=256, batch_size=16,
                     max_windows=200):
    """Score the *same* target characters with long vs short visible history.

    Targets ids[s+short+1 : s+long+1] are predicted twice: once at positions
    short..long-1 of a `long`-character window (so the model has seen at least
    `short` characters), and once at positions 0..short-1 of a fresh window
    starting at s+short (so it has seen almost nothing). Identical targets,
    identical model, only the available context differs.
    """
    starts = list(range(0, len(ids) - long - 1, long))[:max_windows]
    long_total = short_total = 0.0
    count = 0

    for i in range(0, len(starts), batch_size):
        batch = starts[i:i + batch_size]

        full = torch.tensor([ids[s:s + long + 1].tolist() for s in batch],
                            dtype=torch.long, device=device)
        logits, _ = model(full[:, :-1])
        targets = full[:, 1:][:, short:]
        long_total += F.cross_entropy(
            logits[:, short:].reshape(-1, logits.size(-1)),
            targets.reshape(-1), reduction="sum").item()

        tail = torch.tensor(
            [ids[s + short:s + long + 1].tolist() for s in batch],
            dtype=torch.long, device=device)
        logits2, _ = model(tail[:, :-1])
        short_total += F.cross_entropy(
            logits2.reshape(-1, logits2.size(-1)),
            tail[:, 1:].reshape(-1), reduction="sum").item()

        count += targets.numel()

    return long_total / count, short_total / count


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, vocab, cfg = load(args.ckpt, device)

    train_ds, val_ds, _, _ = build_datasets(
        cfg["corpus"], seq_len=cfg["seq_len"], min_freq=cfg["min_freq"],
        val_fraction=cfg["val_fraction"])
    train_ids = train_ds.ids.numpy()
    val_ids = val_ds.ids.numpy()

    V = len(vocab)
    rows = [
        ("uniform over vocab", math.log(V)),
        ("unigram, fitted on train", unigram_nll(train_ids, val_ids, V)),
        ("unigram, oracle (val entropy)", unigram_entropy(val_ids)),
        ("this LSTM LM", corpus_nll(model, val_ids, device,
                                    seq_len=args.seq_len,
                                    max_windows=args.max_windows)),
    ]

    # shuffling keeps the unigram distribution and destroys only the ordering
    generator = torch.Generator().manual_seed(0)
    perm = torch.randperm(len(val_ids), generator=generator).numpy()
    rows.append(("this LSTM LM, shuffled text",
                 corpus_nll(model, val_ids[perm], device,
                            seq_len=args.seq_len,
                            max_windows=args.max_windows)))

    print(f"vocab size: {V}   held-out chars: {len(val_ids):,}   "
          f"context: {args.seq_len}\n")
    print(f"{'model':<32} {'nats/char':>10} {'bits/char':>10} {'ppl':>10}")
    print("-" * 64)
    for name, nats in rows:
        print(f"{name:<32} {nats:>10.4f} {bits_per_char(nats):>10.3f} "
              f"{perplexity(nats):>10.2f}")

    by_name = dict(rows)
    model_nats = by_name["this LSTM LM"]
    print(f"\ngain over train-fitted unigram: "
          f"{(by_name['unigram, fitted on train'] - model_nats) / LN2:.3f} bits/char")
    print(f"gain over oracle unigram:       "
          f"{(by_name['unigram, oracle (val entropy)'] - model_nats) / LN2:.3f} bits/char")
    print(f"penalty on shuffled text:       "
          f"{(by_name['this LSTM LM, shuffled text'] - model_nats) / LN2:.3f} bits/char")

    long_nats, short_nats = context_ablation(
        model, val_ids, device, short=args.short_ctx, long=args.long_ctx,
        max_windows=args.ablation_windows)
    print(f"\ncontext ablation on identical targets "
          f"({args.long_ctx - args.short_ctx:,} chars per window)")
    print(f"  >= {args.short_ctx} chars of history: {long_nats:.4f} nats/char "
          f"({bits_per_char(long_nats):.3f} bits, ppl {perplexity(long_nats):.2f})")
    print(f"  <  {args.short_ctx} chars of history: {short_nats:.4f} nats/char "
          f"({bits_per_char(short_nats):.3f} bits, ppl {perplexity(short_nats):.2f})")
    print(f"  difference: {(short_nats - long_nats) / LN2:.3f} bits/char")

    if args.save:
        payload = {name: {"nats": n, "bits": bits_per_char(n),
                          "ppl": perplexity(n)} for name, n in rows}
        payload["parameters"] = model.num_parameters()
        payload["context"] = args.seq_len
        with open(args.save, "w", encoding="utf-8") as fw:
            json.dump(payload, fw, ensure_ascii=False, indent=2)
        print(f"\nwrote {args.save}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="runs/best.pt")
    parser.add_argument("--seq_len", type=int, default=256,
                        help="evaluation context; use 128 to match the "
                             "Transformer's maximum context")
    parser.add_argument("--max_windows", type=int, default=400)
    parser.add_argument("--short_ctx", type=int, default=128)
    parser.add_argument("--long_ctx", type=int, default=256)
    parser.add_argument("--ablation_windows", type=int, default=200)
    parser.add_argument("--save", type=str, default=None)
    main(parser.parse_args())
