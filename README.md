# Character-level LSTM language model (Chinese)

A character-level language model trained from scratch on 5.98M characters of
Chinese text. The LSTM recurrence is implemented by hand — `nn.LSTM` is not
used anywhere.

It grew out of a coursework assignment. CS324 Deep Learning (SUSTech, Fall
2024), Assignment 3 Part I required implementing an LSTM from scratch,
explicitly without `torch.nn.LSTM`, deriving all four gates and unrolling the
recurrence manually. That assignment trained the network on `PalindromeDataset`,
a synthetic task: given the first T−1 digits of a digit palindrome, predict the
T-th. **The coursework itself is not published here** — the assignment
specification and my submission are available on request.

## Why the extension exists

The assignment's objective (equation 9 of its specification) computes
cross-entropy at the **final timestep only**:

```
L = - sum_k  y_k log( y~_k^(T) )          <- note the superscript (T)
```

The network reads a sequence, emits one label, and the intermediate hidden
states receive no supervision of their own. That is sequence classification.

A language model instead predicts the next token at **every** position:

```
L = - (1/T) sum_t sum_k  y_k^(t) log( y~_k^(t) )
```

One superscript is the entire boundary between the two. Crossing it is not a
local edit: a target at every step means the data must be real text rather than
synthetic labels, which means a vocabulary, which means an embedding table on
the input side and an output projection applied at every timestep.

## What changed

| | assignment | this |
|---|---|---|
| Data | synthetic digit palindromes | 5.98M characters of Chinese text |
| Vocabulary | 10 digits | 4,955 characters (min_freq 2, `<unk>` for the tail) |
| Input | one-hot / raw scalar | learned embedding, 256-d |
| Objective | cross-entropy at the **last** step | next-char cross-entropy at **every** step |
| Depth | 1 layer | 2 layers + dropout |
| Size | ~68K parameters | 7,484,507 parameters |
| Optimiser | RMSProp, fixed LR | AdamW, cosine decay + warmup |
| Metric | accuracy | perplexity / bits-per-char |
| Output | a class label | autoregressive generation |

The gate equations are unchanged:

```
g_t = tanh   (W_gx x_t + W_gh h_{t-1})
i_t = sigmoid(W_ix x_t + W_ih h_{t-1})
f_t = sigmoid(W_fx x_t + W_fh h_{t-1})
o_t = sigmoid(W_ox x_t + W_oh h_{t-1})
c_t = g_t * i_t + c_{t-1} * f_t
h_t = tanh(c_t) * o_t
```

The assignment's eight `nn.Linear` layers are fused into two (`Wx: input -> 4H`,
`Wh: hidden -> 4H`, then `chunk(4)`) — mathematically identical, ~4x faster,
which matters at 6M characters instead of a toy set.

## Results

5,000 steps, ~26 minutes on one RTX 5060 Laptop. Held-out perplexity **69.2**
(6.11 bits/char).

| | bits/char | perplexity |
|---|---|---|
| uniform over vocabulary | 12.275 | 4955.00 |
| unigram, fitted on train | 9.530 | 739.28 |
| unigram, oracle (entropy of val) | 9.393 | 672.25 |
| **this model** | **6.011** | **64.49** |
| this model, on shuffled text | 11.989 | 4065.07 |

Shuffling the held-out characters preserves the unigram distribution exactly
and destroys only ordering; the model loses 5.98 bits/char and falls back near
the uniform baseline, so essentially all of its advantage is sequential
structure rather than memorised character frequencies.

A controlled ablation on identical target characters puts the benefit of ≥128
characters of visible history at 0.049 bits/char — real, but much smaller than
an uncontrolled 128-vs-256-window comparison suggests.

Sample at step 5000 (temperature 0.8, top-k 40):

```
据新华社伦敦１月１日电（记者黄建）在香港特区政府总理李岚清（附图片１张）
```

The news-wire dateline template is reproduced correctly; the office it names
does not exist. Full write-up — structure, settings, curves, samples, baselines,
ablation and analysis — in [`REPORT.md`](REPORT.md).

## Files

| file | contents |
|---|---|
| `data.py` | corpus loading, character vocabulary, next-char windowing |
| `model.py` | hand-written `LSTMLayer` + `CharLSTMLM`, sampling |
| `train.py` | training loop, evaluation, checkpointing, loss curves |
| `eval.py` | unigram baselines, shuffle control, context-length ablation |
| `sample.py` | generation from a checkpoint |
| `utils.py` | `AverageMeter` (carried over from the assignment), metrics, LR schedule |

## Data

The corpus is **not redistributed here**. The runs in `REPORT.md` used two
word-segmented Chinese corpora (5,982,899 characters combined after
preprocessing, 17.7 MB of UTF-8) that are not mine to publish.

Any UTF-8 text works. `data.py` builds the vocabulary from whatever it is given
and strips spaces by default, so word-segmented corpora and plain running text
behave the same. `DEFAULT_CORPUS` in `train.py` points at local paths on the
machine the runs were done on; pass `--corpus` instead of relying on it.

## Setup

```bash
pip install -r requirements.txt

python train.py --corpus your_text.txt --max_steps 5000   # ~26 min on an RTX 5060
python eval.py  --ckpt runs/best.pt
python sample.py --prompt "中国经济" --n 300 --temperature 0.8 --top_k 40
```

A 100-step smoke run to check the pipeline end to end:

```bash
python train.py --corpus your_text.txt --max_steps 100 \
    --eval_interval 50 --sample_interval 50 --out_dir runs_smoke
```

Step-0 training loss should print as `ln(vocab_size)` — 8.5081 against a
theoretical 8.5082 for the 4,955-character vocabulary used here. If it does
not, something upstream of the optimiser is wrong.

Default architecture: 2 layers, hidden 512, embedding 256, dropout 0.2,
sequence length 128, AdamW with cosine decay and 200 warmup steps, gradient
clipping at 1.0, bf16 autocast.

## Design choices worth noting

- **Contiguous train/val split, not random.** Splitting *windows* at random
  would put nearly every validation character inside some training window,
  since windows are adjacent in the underlying text. The raw text is split
  first and windows built independently from each half.
- **Segmentation spaces stripped.** The source corpus is word-segmented. Left
  in, the space would be ~30% of all tokens and the model would spend much of
  its capacity predicting word boundaries.
- **Forget-gate bias initialised to 1**, so the cell state is retained rather
  than erased during the first few hundred updates.
- **Two unigram baselines.** The fitted one is what a unigram model actually
  achieves; the oracle one is the entropy of the validation distribution
  itself, which by Gibbs' inequality lower-bounds any unigram model on this
  text and is therefore the stricter comparison.
- **Perplexity and bits-per-char, not accuracy.** Next-character accuracy is
  close to meaningless for a language model — the correct next character is
  genuinely ambiguous most of the time.

## Limitations

- An LSTM at 7.5M parameters on 5.98M characters: roughly four orders of
  magnitude below a modern language model. Characters rather than a learned
  subword tokenizer, no attention.
- One training run, one seed. The context-length ablation is the only
  controlled comparison; no hyperparameter sweep, no scaling curve.
- Perplexity is not compared against any published benchmark, since the corpus
  is not a standard one.

It is an exercise in the mechanics of language-model training — the next-token
objective, perplexity and bits-per-char, truncated BPTT, autoregressive
sampling, and honest baselines — not a competitive model.
