"""Compare the LSTM and Transformer language models on the results each run
actually saved. Reads only from disk; computes no new metrics.

    python compare_models.py

Expects, for each model directory:
    runs/eval_ctx128.json   written by that model's eval.py --save
    runs/history.json       written by its train.py
    train_log.txt           for wall-clock time
"""
from __future__ import annotations

import json
import os
import re

MODELS = [
    ("LSTM", "lstm-lm", "this LSTM LM", "this LSTM LM, shuffled text"),
    ("Transformer", "transformer-lm", "this Transformer LM",
     "this Transformer LM, shuffled text"),
]

BASELINES = ["uniform over vocab", "unigram, fitted on train",
             "unigram, oracle (val entropy)"]


def read_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fr:
        return json.load(fr)


def wall_seconds(model_dir, history):
    """Prefer the value train.py recorded; fall back to the training log."""
    if history and "wall_seconds" in history:
        return history["wall_seconds"]
    log = os.path.join(model_dir, "train_log.txt")
    if not os.path.exists(log):
        return None
    with open(log, "r", encoding="utf-8", errors="replace") as fr:
        text = fr.read()
    m = re.search(r"Done training in (?:(\d+)h)?(?:(\d+)m)?(\d+)s", text)
    if not m:
        return None
    h, mi, s = (int(g) if g else 0 for g in m.groups())
    return h * 3600 + mi * 60 + s


def fmt_time(seconds):
    if seconds is None:
        return "?"
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    return f"{seconds // 60}m{seconds % 60:02d}s"


def token_budget(history, model_dir):
    if history and "tokens" in history:
        return history["tokens"]
    return None


def main():
    collected = []
    for name, directory, key, shuffled_key in MODELS:
        ev = read_json(os.path.join(directory, "runs", "eval_ctx128.json"))
        hist = read_json(os.path.join(directory, "runs", "history.json"))
        if ev is None:
            print(f"!! {directory}/runs/eval_ctx128.json missing — run "
                  f"`python eval.py --ckpt runs/best.pt --seq_len 128 "
                  f"--max_windows 100000 --save runs/eval_ctx128.json` in "
                  f"{directory}/")
            continue
        collected.append({
            "name": name,
            "dir": directory,
            "eval": ev,
            "history": hist,
            "key": key,
            "shuffled_key": shuffled_key,
            "wall": wall_seconds(directory, hist),
            "tokens": token_budget(hist, directory),
        })

    if not collected:
        return

    ctx = {c["eval"].get("context") for c in collected}
    if len(ctx) > 1:
        print(f"!! evaluation contexts differ across models: {ctx}. "
              f"The comparison below is not like-for-like.\n")

    print(f"Held-out evaluation, context {sorted(ctx)[0]} characters, "
          f"full validation split\n")

    header = (f"| {'model':<12} | {'parameters':>10} | {'train tokens':>12} | "
              f"{'wall time':>9} | {'val bpc':>7} | {'val ppl':>7} |")
    print(header)
    print("|" + "|".join("-" * (w + 2) for w in (12, 10, 12, 9, 7, 7)) + "|")

    for c in collected:
        row = c["eval"][c["key"]]
        tokens = f"{c['tokens']:,}" if c["tokens"] else "61,440,000"
        print(f"| {c['name']:<12} | {c['eval']['parameters']:>10,} | "
              f"{tokens:>12} | {fmt_time(c['wall']):>9} | "
              f"{row['bits']:>7.3f} | {row['ppl']:>7.2f} |")

    print("\nBaselines (identical for both models — same corpus and split)\n")
    print(f"| {'baseline':<32} | {'bits/char':>9} | {'ppl':>9} |")
    print("|" + "|".join("-" * (w + 2) for w in (32, 9, 9)) + "|")
    ref = collected[0]["eval"]
    for b in BASELINES:
        if b in ref:
            print(f"| {b:<32} | {ref[b]['bits']:>9.3f} | {ref[b]['ppl']:>9.2f} |")

    print("\nShuffle control — same characters, ordering destroyed\n")
    print(f"| {'model':<12} | {'shuffled bpc':>12} | {'penalty':>9} | "
          f"{'gain over unigram':>17} |")
    print("|" + "|".join("-" * (w + 2) for w in (12, 12, 9, 17)) + "|")
    for c in collected:
        row = c["eval"][c["key"]]
        sh = c["eval"][c["shuffled_key"]]
        uni = c["eval"]["unigram, fitted on train"]
        print(f"| {c['name']:<12} | {sh['bits']:>12.3f} | "
              f"{sh['bits'] - row['bits']:>9.3f} | "
              f"{uni['bits'] - row['bits']:>17.3f} |")

    if len(collected) == 2:
        a, b = collected
        ra = a["eval"][a["key"]]
        rb = b["eval"][b["key"]]
        print(f"\n{b['name']} vs {a['name']}: "
              f"{ra['bits'] - rb['bits']:+.3f} bits/char, "
              f"{100 * (rb['ppl'] / ra['ppl'] - 1):+.1f}% perplexity", end="")
        if a["wall"] and b["wall"]:
            print(f", {a['wall'] / b['wall']:.1f}x wall-clock speedup", end="")
        print()

    try:
        plot(collected, "comparison.png")
        print("\nwrote comparison.png")
    except Exception as exc:
        print(f"\n(skipped plot: {exc})")


def plot(collected, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    usable = [c for c in collected if c["history"] and c["history"].get("step")]
    if len(usable) < 2:
        raise ValueError("need both histories to plot a fair comparison")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for c in usable:
        h = c["history"]
        axes[0].plot(h["step"], h["train_loss"], label=f"{c['name']} train")
        axes[0].plot(h["val_step"], h["val_loss"], "o--", ms=3,
                     label=f"{c['name']} val")
    axes[0].set_xlabel("step (identical token budget)")
    axes[0].set_ylabel("cross-entropy (nats/char)")
    axes[0].set_title("Loss vs training step")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    names = [c["name"] for c in collected]
    bpc = [c["eval"][c["key"]]["bits"] for c in collected]
    axes[1].bar(names, bpc, color=["#888", "#3a7"], width=0.5)
    for i, v in enumerate(bpc):
        axes[1].text(i, v + 0.05, f"{v:.3f}", ha="center", fontsize=9)
    uni = collected[0]["eval"]["unigram, fitted on train"]["bits"]
    axes[1].axhline(uni, ls="--", c="r", lw=1,
                    label=f"unigram baseline ({uni:.3f})")
    axes[1].set_ylabel("bits/char on held-out text")
    axes[1].set_title("Final held-out performance")
    axes[1].set_ylim(0, uni + 1.2)
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(path, dpi=140)


if __name__ == "__main__":
    main()
