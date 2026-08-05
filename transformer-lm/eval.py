"""Score the Transformer against unigram baselines and a shuffled copy of the
held-out text, using the same methodology as ../lstm-lm/eval.py.

    python eval.py --ckpt runs/best.pt
"""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter

import torch
import torch.nn.functional as F

from data import build_datasets
from sample import load
from utils import bits_per_char, perplexity

LN2 = math.log(2.0)


@torch.no_grad()
def corpus_nll(model, ids, device, seq_len=128, batch_size=32, max_windows=None):
    """Mean next-char cross-entropy over a flat id array.

    seq_len defaults to the model's training context; unlike the LSTM the
    Transformer cannot be evaluated beyond max_len without extrapolating the
    learned positional embeddings, so windows are capped there.
    """
    seq_len = min(seq_len, model.max_len)
    windows = []
    for start in range(0, len(ids) - seq_len - 1, seq_len):
        windows.append(ids[start:start + seq_len + 1])
        if max_windows and len(windows) >= max_windows:
            break
    if not windows:
        raise ValueError("text too short to evaluate")

    total, count = 0.0, 0
    for i in range(0, len(windows), batch_size):
        chunk = torch.tensor([w.tolist() for w in windows[i:i + batch_size]],
                             dtype=torch.long, device=device)
        logits = model(chunk[:, :-1])
        total += F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                 chunk[:, 1:].reshape(-1),
                                 reduction="sum").item()
        count += chunk[:, 1:].numel()
    return total / count


def unigram_nll(train_ids, val_ids, vocab_size, alpha=1.0):
    counts = Counter(int(i) for i in train_ids)
    denom = len(train_ids) + alpha * vocab_size
    logp = [math.log((counts.get(t, 0) + alpha) / denom) for t in range(vocab_size)]
    return -sum(logp[int(i)] for i in val_ids) / len(val_ids)


def unigram_entropy(ids):
    counts = Counter(int(i) for i in ids)
    n = sum(counts.values())
    return -sum((c / n) * math.log(c / n) for c in counts.values())


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, vocab, cfg = load(args.ckpt, device)

    train_ds, val_ds, _, _ = build_datasets(
        cfg["corpus"], seq_len=cfg["seq_len"], min_freq=cfg["min_freq"],
        val_fraction=cfg["val_fraction"])
    train_ids = train_ds.ids.numpy()
    val_ids = val_ds.ids.numpy()

    V = len(vocab)
    model_nats = corpus_nll(model, val_ids, device, seq_len=args.seq_len,
                            max_windows=args.max_windows)

    generator = torch.Generator().manual_seed(0)
    perm = torch.randperm(len(val_ids), generator=generator).numpy()
    shuffled_nats = corpus_nll(model, val_ids[perm], device,
                               seq_len=args.seq_len,
                               max_windows=args.max_windows)

    rows = [
        ("uniform over vocab", math.log(V)),
        ("unigram, fitted on train", unigram_nll(train_ids, val_ids, V)),
        ("unigram, oracle (val entropy)", unigram_entropy(val_ids)),
        ("this Transformer LM", model_nats),
        ("this Transformer LM, shuffled text", shuffled_nats),
    ]

    print(f"vocab size: {V}   held-out chars: {len(val_ids):,}   "
          f"context: {min(args.seq_len, model.max_len)}\n")
    print(f"{'model':<36} {'nats/char':>10} {'bits/char':>10} {'ppl':>10}")
    print("-" * 68)
    for name, nats in rows:
        print(f"{name:<36} {nats:>10.4f} {bits_per_char(nats):>10.3f} "
              f"{perplexity(nats):>10.2f}")

    by_name = dict(rows)
    print(f"\ngain over train-fitted unigram: "
          f"{(by_name['unigram, fitted on train'] - model_nats) / LN2:.3f} bits/char")
    print(f"penalty on shuffled text:       "
          f"{(shuffled_nats - model_nats) / LN2:.3f} bits/char")
    print(f"parameters: {model.num_parameters():,}")

    if args.save:
        payload = {name: {"nats": n, "bits": bits_per_char(n),
                          "ppl": perplexity(n)} for name, n in rows}
        payload["parameters"] = model.num_parameters()
        payload["context"] = min(args.seq_len, model.max_len)
        with open(args.save, "w", encoding="utf-8") as fw:
            json.dump(payload, fw, ensure_ascii=False, indent=2)
        print(f"wrote {args.save}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="runs/best.pt")
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--max_windows", type=int, default=400)
    parser.add_argument("--save", type=str, default=None)
    main(parser.parse_args())
