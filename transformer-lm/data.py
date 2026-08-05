"""Corpus loading and next-character windowing."""
from __future__ import annotations

import io
import os
from collections import Counter

import numpy as np
import torch
import torch.utils.data as data


class CharVocab:
    """Character to id, with rare characters folded into <unk>."""

    UNK = "<unk>"

    def __init__(self, text: str, min_freq: int = 2):
        counter = Counter(text)
        kept = [ch for ch, n in counter.most_common() if n >= min_freq]
        self.itos = [self.UNK] + kept
        self.stoi = {ch: i for i, ch in enumerate(self.itos)}
        self.counter = counter

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> np.ndarray:
        unk = 0
        stoi = self.stoi
        return np.fromiter((stoi.get(ch, unk) for ch in text), dtype=np.int32,
                           count=len(text))

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)

    def save(self, path: str) -> None:
        with io.open(path, "w", encoding="utf-8") as fw:
            fw.write("\n".join(repr(ch) for ch in self.itos))

    @classmethod
    def from_itos(cls, itos):
        obj = cls.__new__(cls)
        obj.itos = list(itos)
        obj.stoi = {ch: i for i, ch in enumerate(obj.itos)}
        obj.counter = Counter()
        return obj


def load_text(paths, strip_spaces: bool = True) -> str:
    chunks = []
    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"corpus file not found: {path}")
        with io.open(path, "r", encoding="utf-8") as fr:
            text = fr.read()
        if strip_spaces:
            # the corpus is word-segmented; spaces would be ~30% of all tokens
            text = text.replace(" ", "").replace("　", "")
        chunks.append(text)
    return "\n".join(chunks)


class CharDataset(data.Dataset):
    """Windows of seq_len ids, with target[t] == input[t + 1]."""

    def __init__(self, ids: np.ndarray, seq_len: int, stride: int | None = None):
        self.ids = torch.from_numpy(ids.astype(np.int64))
        self.seq_len = seq_len
        self.stride = stride or seq_len
        self.n = max(0, (len(self.ids) - seq_len - 1) // self.stride + 1)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        start = idx * self.stride
        chunk = self.ids[start:start + self.seq_len + 1]
        return chunk[:-1], chunk[1:]


def build_datasets(paths, seq_len: int, min_freq: int = 2,
                   val_fraction: float = 0.02, strip_spaces: bool = True):
    text = load_text(paths, strip_spaces=strip_spaces)

    # split the raw text, not the windows -- windows are adjacent in the
    # underlying text, so splitting those at random leaks nearly all of val
    split_at = int(len(text) * (1.0 - val_fraction))
    train_text, val_text = text[:split_at], text[split_at:]

    vocab = CharVocab(train_text, min_freq=min_freq)
    train_ds = CharDataset(vocab.encode(train_text), seq_len)
    val_ds = CharDataset(vocab.encode(val_text), seq_len)
    return train_ds, val_ds, vocab, text
