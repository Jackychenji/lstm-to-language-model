# Character-level causal Transformer

A decoder-only Transformer language model, trained to be compared fairly with
the hand-written LSTM in [`../lstm-lm`](../lstm-lm): same corpus, same
vocabulary, same split, same 61,440,000-token budget, same evaluation, matched
parameter count.

After the original CS324 assignment had been completed and graded, I
independently extended the handwritten LSTM into a character-level language
model; this Transformer is the follow-up comparison. Neither is graded
coursework.

## Architecture

```
token embedding (4955 -> 256) + learned positional embedding (128 -> 256)
6 x [ x + Attn(LayerNorm(x)) ; x + FFN(LayerNorm(x)) ]     # pre-LN
LayerNorm
Linear 256 -> 4955                                          # untied
```

d_model 256, 6 layers, 8 heads, d_ff 1024, dropout 0.2, max context 128.
**7,313,755 parameters**, 2.3% fewer than the LSTM's 7,484,507.

Blocks are written out rather than using `nn.TransformerEncoder`; the attention
kernel is `F.scaled_dot_product_attention(..., is_causal=True)`. No KV cache —
generation recomputes over the trailing 128 characters.

## Results

5,000 steps in **4m32s** on one RTX 5060 Laptop. Held-out perplexity **41.49**
(5.375 bits/char) over the full validation split at 128-character context.

| | bits/char | perplexity |
|---|---|---|
| uniform over vocabulary | 12.275 | 4955.00 |
| unigram, fitted on train | 9.530 | 739.28 |
| LSTM (`../lstm-lm`) | 6.113 | 69.22 |
| **this model** | **5.375** | **41.49** |
| this model, on shuffled text | 12.908 | 7686.09 |

Against the LSTM at the same token budget: **0.738 bits/char better, 40.1%
lower perplexity**, with 2.3% fewer parameters. The training run was also 5.7x
faster, but that figure needs a caveat the perplexity figure does not.

The trained LSTM unrolls its recurrence in a Python loop while this model calls
a fused attention kernel, so 5.7x mixes architecture with implementation.
`../benchmark_speed.py` adds cuDNN's fused `nn.LSTM` as a timing reference:
hand-written loop → fused `nn.LSTM` is **5.99x** (5.54–6.22), while fused
`nn.LSTM` → Transformer is **1.01x** (0.88–1.07), a range straddling 1.0. Most
of the observed gap comes from kernel and implementation differences rather
than from architecture alone. **Perplexity, not training time, is the
meaningful architecture result.**

The shuffle row is the more interesting one. Shuffling the held-out characters
preserves the unigram distribution and destroys only ordering; it costs this
model 7.533 bits/char against the LSTM's 5.910, and pushes it *past* the
uniform baseline — confidently wrong rather than merely uninformed. Both models
learned ordering; this one committed harder to it.

Data provenance, hashes and preprocessing: [`../DATA.md`](../DATA.md).

Full write-up, including what is and is not controlled in the comparison:
[`REPORT.md`](REPORT.md). Side-by-side tables: `python compare_models.py` from
the repository root.

## Setup

```bash
pip install -r requirements.txt

python train.py --corpus your_text.txt --max_steps 5000   # ~4.5 min on an RTX 5060
python eval.py --ckpt runs/best.pt --seq_len 128 --max_windows 100000 \
    --save runs/eval_ctx128.json
python sample.py --prompt "中国经济" --n 300 --temperature 0.8 --top_k 40
```

If batch 96 does not fit in VRAM, use gradient accumulation to keep the
effective batch identical:

```bash
python train.py --batch_size 96 --micro_batch 32   # 3 accumulation steps
```

Step-0 training loss should print near `ln(vocab_size)` — 8.5631 against 8.5082
here. The excess over the exact value is the randomly initialised output
projection, which starts with slightly non-uniform logits.

## Caveats worth reading before quoting the comparison

- The learning rates differ (6e-4 here, 2e-3 for the LSTM). Each is a
  conventional default for its architecture, neither was tuned. This is a
  comparison of two architectures at reasonable settings, not a
  single-variable experiment.
- "Sequence length 128" means direct attention to all earlier positions here,
  and compression into a fixed-width cell state for the LSTM. Same number,
  different mechanism — which is the thing being measured.
- One seed each, no hyperparameter sweep, no scaling curve.
- The direction of the result matches the published literature. What is worth
  anything here is that data, vocabulary, token budget and evaluation are
  actually held fixed, which is where informal architecture comparisons
  usually go wrong.
