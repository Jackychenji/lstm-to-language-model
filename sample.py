"""Generate text from a trained checkpoint.

    python sample.py --prompt "中国经济" --n 300 --temperature 0.8 --top_k 40
"""
from __future__ import annotations

import argparse
import io
import sys

import torch

from data import CharVocab
from model import CharLSTMLM


def load(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    vocab = CharVocab.from_itos(ckpt["itos"])
    model = CharLSTMLM(len(vocab), emb_dim=cfg["emb_dim"],
                       hidden_dim=cfg["hidden_dim"],
                       num_layers=cfg["num_layers"],
                       dropout=0.0).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, vocab, cfg


@torch.no_grad()
def nll(model, vocab, text, device):
    """Cross-entropy the model assigns to a given string."""
    ids = torch.tensor(vocab.encode(text).astype("int64"),
                       device=device).unsqueeze(0)
    logits, _ = model(ids[:, :-1])
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), ids[:, 1:].reshape(-1))
    return loss.item()


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, vocab, cfg = load(args.ckpt, device)

    out = sys.stdout
    if args.out:
        out = io.open(args.out, "w", encoding="utf-8")

    for i in range(args.num_samples):
        ids = vocab.encode(args.prompt).tolist()
        generated = model.generate(ids, args.n, temperature=args.temperature,
                                   top_k=args.top_k, device=device)
        out.write(f"--- sample {i + 1} (T={args.temperature}, "
                  f"top_k={args.top_k}) ---\n")
        out.write(vocab.decode(generated) + "\n\n")

    if args.score:
        print(f"\nNLL of prompt continuation: {nll(model, vocab, args.score, device):.4f}",
              file=sys.stderr)

    if args.out:
        out.close()
        print(f"wrote {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="runs/best.pt")
    parser.add_argument("--prompt", type=str, default="中国")
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--score", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)
    main(parser.parse_args())
