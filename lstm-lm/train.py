"""Train the character-level LSTM language model.

    python train.py --max_steps 5000
"""
from __future__ import absolute_import, division, print_function

import argparse
import json
import os
import time

import torch
from torch.utils.data import DataLoader

from data import build_datasets
from model import CharLSTMLM
from utils import AverageMeter, bits_per_char, cosine_lr, format_time, perplexity

DEFAULT_CORPUS = [
    "../../../laptopFile/shortestPathWordSegmentation/dataset/dataset1_training.utf8",
    "../../../laptopFile/shortestPathWordSegmentation/dataset/dataset2_training.utf8",
]

SAMPLE_PROMPTS = ["中国", "他说", "经济", "这个"]


def infinite(loader):
    while True:
        for batch in loader:
            yield batch


@torch.no_grad()
def evaluate(model, loader, criterion, device, max_batches=None):
    model.eval()
    losses = AverageMeter("ValLoss", fmt=":.4f")
    for step, (inputs, targets) in enumerate(loader):
        if max_batches is not None and step >= max_batches:
            break
        inputs, targets = inputs.to(device), targets.to(device)
        logits, _ = model(inputs)
        loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        losses.update(loss.item(), inputs.size(0))
    model.train()
    return losses.avg


def sample_text(model, vocab, device, prompts=SAMPLE_PROMPTS, n_tokens=120,
                temperature=0.8, top_k=40):
    lines = []
    for prompt in prompts:
        prompt_ids = vocab.encode(prompt).tolist()
        out = model.generate(prompt_ids, n_tokens, temperature=temperature,
                             top_k=top_k, device=device)
        text = vocab.decode(out).replace("\n", " / ")
        lines.append(f"[{prompt}] {text}")
    model.train()
    return lines


def main(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(config.out_dir, exist_ok=True)
    torch.manual_seed(config.seed)

    print(f"device: {device}")
    if device == "cuda":
        print(f"gpu:    {torch.cuda.get_device_name(0)}")

    # ---------------------------------------------------------------- data
    t0 = time.time()
    train_ds, val_ds, vocab, text = build_datasets(
        config.corpus, seq_len=config.seq_len, min_freq=config.min_freq,
        val_fraction=config.val_fraction)
    print(f"corpus: {len(text):,} chars | vocab: {len(vocab):,} | "
          f"train windows: {len(train_ds):,} | val windows: {len(val_ds):,} "
          f"({time.time() - t0:.1f}s)")

    train_loader = DataLoader(train_ds, batch_size=config.batch_size,
                              shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size)

    # --------------------------------------------------------------- model
    model = CharLSTMLM(len(vocab), emb_dim=config.emb_dim,
                       hidden_dim=config.hidden_dim,
                       num_layers=config.num_layers,
                       dropout=config.dropout).to(device)
    print(f"params: {model.num_parameters():,}")

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay,
                                  betas=(0.9, 0.95))

    use_amp = bool(device == "cuda" and config.amp)
    amp_dtype = torch.bfloat16

    # ---------------------------------------------------------------- loop
    history = {"step": [], "train_loss": [], "val_step": [], "val_loss": []}
    sample_log = open(os.path.join(config.out_dir, "samples.txt"), "w",
                      encoding="utf-8")
    best_val = float("inf")
    running = AverageMeter("Loss", fmt=":.4f")
    batches = infinite(train_loader)
    start = time.time()

    model.train()
    for step in range(config.max_steps):
        lr = cosine_lr(step, config.max_steps, config.learning_rate,
                       config.warmup_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr

        inputs, targets = next(batches)
        inputs, targets = inputs.to(device), targets.to(device)

        with torch.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
            logits, _ = model(inputs)
            loss = criterion(logits.reshape(-1, logits.size(-1)).float(),
                             targets.reshape(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_norm)
        optimizer.step()

        running.update(loss.item(), inputs.size(0))

        if step % config.log_interval == 0:
            elapsed = time.time() - start
            done = step + 1
            eta = elapsed / done * (config.max_steps - done)
            print(f"[{step:>5}/{config.max_steps}] loss {running.avg:.4f} | "
                  f"ppl {perplexity(running.avg):7.2f} | "
                  f"bpc {bits_per_char(running.avg):.3f} | "
                  f"lr {lr:.2e} | {format_time(elapsed)} elapsed, "
                  f"eta {format_time(eta)}")
            history["step"].append(step)
            history["train_loss"].append(running.avg)
            running.reset()

        if step > 0 and step % config.eval_interval == 0:
            val_loss = evaluate(model, val_loader, criterion, device,
                                max_batches=config.eval_batches)
            history["val_step"].append(step)
            history["val_loss"].append(val_loss)
            tag = ""
            if val_loss < best_val:
                best_val = val_loss
                tag = "  <- best, saved"
                torch.save({"model": model.state_dict(),
                            "itos": vocab.itos,
                            "config": vars(config)},
                           os.path.join(config.out_dir, "best.pt"))
            print(f"  >> val loss {val_loss:.4f} | ppl {perplexity(val_loss):.2f} "
                  f"| bpc {bits_per_char(val_loss):.3f}{tag}")

        if step > 0 and step % config.sample_interval == 0:
            lines = sample_text(model, vocab, device)
            print(f"  -- samples @ step {step}")
            for line in lines:
                print(f"     {line}")
                sample_log.write(f"step {step}\t{line}\n")
            sample_log.flush()

    # -------------------------------------------------------------- finish
    val_loss = evaluate(model, val_loader, criterion, device,
                        max_batches=config.eval_batches)
    history["val_step"].append(config.max_steps)
    history["val_loss"].append(val_loss)
    if val_loss < best_val:
        best_val = val_loss
        torch.save({"model": model.state_dict(), "itos": vocab.itos,
                    "config": vars(config)},
                   os.path.join(config.out_dir, "best.pt"))

    print(f"\nDone training in {format_time(time.time() - start)}.")
    print(f"final val loss {val_loss:.4f} | ppl {perplexity(val_loss):.2f} | "
          f"bpc {bits_per_char(val_loss):.3f}")
    print(f"best  val loss {best_val:.4f} | ppl {perplexity(best_val):.2f}")

    print("\nFinal samples:")
    for line in sample_text(model, vocab, device, n_tokens=180):
        print(f"  {line}")
        sample_log.write(f"final\t{line}\n")
    sample_log.close()

    with open(os.path.join(config.out_dir, "history.json"), "w",
              encoding="utf-8") as fw:
        json.dump(history, fw, ensure_ascii=False, indent=2)

    try:
        plot_history(history, os.path.join(config.out_dir, "loss_curve.png"))
        print(f"\nwrote {config.out_dir}/loss_curve.png")
    except Exception as exc:  # matplotlib is optional
        print(f"(skipped loss curve: {exc})")

    return history


def plot_history(history, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history["step"], history["train_loss"], label="train")
    axes[0].plot(history["val_step"], history["val_loss"], "o-", label="val")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("cross-entropy (nats/char)")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history["step"], [perplexity(l) for l in history["train_loss"]],
                 label="train")
    axes[1].plot(history["val_step"], [perplexity(l) for l in history["val_loss"]],
                 "o-", label="val")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("perplexity")
    axes[1].set_title("Perplexity")
    axes[1].set_yscale("log")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=140)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument("--corpus", nargs="+", default=DEFAULT_CORPUS)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--min_freq", type=int, default=2)
    parser.add_argument("--val_fraction", type=float, default=0.02)
    # model
    parser.add_argument("--emb_dim", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    # optimisation
    parser.add_argument("--batch_size", type=int, default=96)
    parser.add_argument("--learning_rate", type=float, default=2e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_norm", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=4000)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--amp", type=int, default=1)
    # bookkeeping
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--eval_batches", type=int, default=40)
    parser.add_argument("--sample_interval", type=int, default=1000)
    parser.add_argument("--out_dir", type=str, default="runs")
    parser.add_argument("--seed", type=int, default=42)

    main(parser.parse_args())
