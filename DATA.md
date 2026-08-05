# Data

The corpus used for the runs reported here is **not redistributed in this
repository**. This file records exactly what was used so the numbers can be
interpreted, and so anyone holding the same files can confirm they have them.

## Source files

Two word-segmented Chinese corpora, space-separated at word boundaries, in
UTF-8. They came to me as `dataset1_training.utf8` and
`dataset2_training.utf8`, bundled with an earlier Chinese word-segmentation
project of mine.

| file | bytes | SHA-256 | characters | lines |
|---|---|---|---|---|
| `dataset1_training.utf8` | 16,891,510 | `872a24b0aa827fe5334370c5d9bf090902a948cc103c4775a3f61285c052e6ee` | 8,705,599 | 86,924 |
| `dataset2_training.utf8` | 7,728,238 | `a45a590f22a226a24d77760e9579fb8339aa2d9500678d43d97b812b60417135` | 4,065,425 | 19,056 |

> **Provenance and licence: not established.** The format — space-delimited
> word segmentation, mainland simplified Chinese, news-wire and literary text
> from the late 1990s — matches the conventions of published Chinese
> segmentation bakeoff corpora, but I have not verified which collection these
> files came from and I am not asserting one. They are therefore not
> redistributed here, and no licence claim is made. If you need the exact data,
> ask and I will point you at what I can establish.

## Preprocessing

`data.py::load_text` concatenates the files with a newline between them and
removes segmentation spaces (`U+0020` and `U+3000`). Nothing else is stripped,
normalised, lowercased or filtered; punctuation, digits, full-width forms and
Latin characters are all kept as ordinary vocabulary items.

| | value |
|---|---|
| Characters after preprocessing | **5,982,899** |
| UTF-8 bytes after preprocessing | **17,725,643** (17.73 MB) |
| Mean bytes per character | 2.96 |
| SHA-256 of the preprocessed text | `326c03aafba03abb5118e085a0bfcb02b460a333efe9b14cd41e74e44c33b084` |

The "17.7 MB" quoted in the reports is that byte count — the UTF-8 encoding of
the concatenated, space-stripped text, not the size of the files on disk
(24.6 MB combined, which still contains the segmentation spaces).

Reproduce with:

```python
import hashlib, sys
sys.path.insert(0, "lstm-lm")
from data import load_text

text = load_text(["dataset1_training.utf8", "dataset2_training.utf8"])
raw = text.encode("utf-8")
print(len(text), len(raw), hashlib.sha256(raw).hexdigest())
```

## Split and vocabulary

The preprocessed text is split **contiguously**, not by shuffling windows: the
first 98% is training text and the last 2% is held out. Windows are then built
independently from each half, so no training window overlaps a validation
window.

| | value |
|---|---|
| Training characters | 5,863,241 |
| Held-out characters | **119,658** |
| Vocabulary | **4,955** symbols, built from the training split only |
| Vocabulary rule | characters appearing ≥ 2 times; rarer folded into `<unk>` |

Both models use this identical vocabulary and split.

## Using your own text

Nothing in the code is specific to this corpus. `--corpus` accepts any UTF-8
file or files:

```bash
python train.py --corpus your_text.txt --max_steps 5000
```

The vocabulary is rebuilt from whatever it is given, so a different corpus
produces a different vocabulary size and the reported perplexities are not
comparable across corpora.
