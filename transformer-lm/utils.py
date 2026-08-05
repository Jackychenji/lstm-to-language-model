"""Training helpers. AverageMeter is carried over from CS324 Assignment 3."""
from __future__ import annotations

import math


class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name}: {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def perplexity(loss: float) -> float:
    return math.exp(min(loss, 20.0))


def bits_per_char(loss: float) -> float:
    return loss / math.log(2.0)


def cosine_lr(step: int, max_steps: int, base_lr: float, warmup: int,
              min_ratio: float = 0.1) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(max_steps - warmup, 1)
    progress = min(max(progress, 0.0), 1.0)
    scale = min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    return base_lr * scale


def format_time(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"
