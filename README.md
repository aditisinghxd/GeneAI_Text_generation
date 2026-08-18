# Text Generation Architecture Comparison: LSTM vs GRU vs FNet

A controlled, character-level comparison of three sequence architectures — LSTM, GRU, and a
causally-adapted FNet — trained on identical data, with the same training loop, optimizer, and
evaluation procedure, so that differences in results can be attributed to the architecture rather than
to inconsistent setup.

## Why this project

Most tutorial-style implementations of these architectures each use different frameworks, tokenization
schemes, datasets, and even different tasks — which makes any "X is better than Y" claim from reading them
side by side essentially meaningless. This project rebuilds all three from a shared foundation so the
comparison is actually valid:

- **One framework** (PyTorch) for all three models
- **One task** — autoregressive next-character prediction
- **One dataset** — character-level, cleaned, identical train/val split
- **One training harness** — same optimizer, same loss, same evaluation procedure, same generation method

## Architectures compared

| Model | Mechanism | Notes |
|---|---|---|
| **LSTM** | Recurrent, 4-gate | Full gating (input/forget/output/candidate), separate cell + hidden state |
| **GRU** | Recurrent, 2-gate | Simplified gating (reset/update), no separate cell state — fewer parameters |
| **FNet (causal)** | Fourier mixing | Replaces self-attention with a 2D FFT for token mixing, applied over a causal sliding window |

### The interesting technical problem: making FNet causal

The original FNet paper applies the Fourier Transform *globally* across the whole input sequence — every
output position is a linear combination of every input position, past **and future**. That's fine for
FNet's original use (a bidirectional encoder, like BERT), but it's fundamentally incompatible with
autoregressive generation: naively applying it to next-token prediction would let the model "see" the
answer baked into its own input representation, producing a model that looks great during training and
fails at real generation.

This project fixes that by applying the Fourier mixing over a fixed-size **causal window** ending at each
position, rather than the whole sequence — position `t` only ever sees positions `[t - window_size + 1,
..., t]`. This keeps FNet's actual mechanic (a fixed Fourier Transform doing the mixing, no learned
attention weights, computed in one vectorized batch operation) while genuinely respecting the
autoregressive constraint.

**This isn't just asserted — it's verified directly in the notebook**: the logits at a given position are
checked before and after mutating tokens *after* that position, confirming a max difference of `0.00e+00`
(exact zero future leakage).

## Dataset

*Alice's Adventures in Wonderland* (Lewis Carroll, public domain, via Project Gutenberg), character-level
tokenization, ~143K cleaned characters, 90/10 train/validation split.

## Methodology

1. **Shared data pipeline** — identical vocabulary, sequence windowing, and batching for all three models
2. **Consistent model interface** — every model exposes `forward(x) -> (logits, hidden)`, so any of the
   three drop into the same training loop unmodified
3. **Sanity checks before training** — each model's untrained loss is checked against the
   random-guessing baseline (`ln(vocab_size)`) to catch data or wiring bugs before spending time training
4. **Shared training/evaluation harness** — same optimizer (Adam), same loss (cross-entropy), same
   gradient clipping, same per-epoch timing and perplexity tracking for all three
5. **Shared generation procedure** — same sliding-context-window generation method used for every model,
   so generated samples are directly comparable

## Results

> Fill in with your actual numbers from the notebook's Part 7 summary table before publishing.

| Model | Parameters | Final Val Loss | Final Val Perplexity | Total Train Time |
|---|---|---|---|---|
| LSTM | `___` | `___` | `___` | `___`s |
| GRU | `___` | `___` | `___` | `___`s |
| FNet (causal) | `___` | `___` | `___` | `___`s |

**Key finding:** `[fill in — e.g. "GRU matched or beat LSTM's validation perplexity with ~23% fewer
parameters and comparable training time, suggesting the simplified gating is sufficient at this model/data
scale."]`

**On FNet:** `[fill in — e.g. "Despite having the fewest parameters, FNet was the slowest to train on CPU,
primarily due to FFT overhead not being as hardware-optimized as cuDNN's recurrent kernels — an
interesting reminder that parameter count alone doesn't determine training cost."]`

Sample generations from all three models (same seed text, same temperature) are in the notebook's Part 6.4
output.

## Project structure

```
.
├── text_generation_comparison.ipynb   # full notebook: data pipeline → models → training → comparison
├── alice.txt                          # cached dataset (downloaded automatically on first run)
├── loss_curves.png                    # training/validation loss plot (generated by the notebook)
├── comparison_summary.png             # parameter/perplexity/time bar charts (generated by the notebook)
└── README.md
```

## How to run

```bash
pip install torch numpy pandas matplotlib jupyter
jupyter notebook text_generation_comparison.ipynb
```

Run all cells top to bottom (Kernel → Restart & Run All) — later sections depend on variables and models
defined earlier. The dataset downloads automatically on first run.

**Runtime:** the default configuration (10 epochs, 250 batches/epoch) takes roughly 15-20 minutes on CPU;
FNet is the slowest of the three due to FFT overhead. For a full-data, more rigorous run, increase
`MAX_BATCHES_PER_EPOCH` to `None` and raise `EPOCHS` — the notebook runs unmodified on a GPU (e.g. Google
Colab's free tier), which substantially closes FNet's speed gap.

## Tech stack

- **PyTorch** — model implementation, training
- **NumPy / Pandas** — data processing, results summarization
- **Matplotlib** — loss curves and comparison charts
- **Jupyter** — development environment

## Limitations & honest caveats

- FNet here is a **causal adaptation**, not the literal architecture from the original paper — the
  original uses global bidirectional mixing and is not designed for autoregressive generation at all. This
  is documented and verified in the notebook rather than glossed over.
- Character-level modeling with a small dataset (~143K characters) and a limited training budget means
  none of the three models produce fluent prose — the comparison is about relative architectural behavior
  under matched conditions, not absolute text quality.
- Results at the default (CPU-friendly) training scale may not generalize to larger data/longer training —
  the notebook's Part 7 discusses this and suggests a full-data GPU run as a natural extension.

## Possible extensions

- Add a small causal **Transformer** (real self-attention) as a fourth baseline — the mechanism FNet was
  originally proposed as a cheaper alternative to
- Word-level or subword (BPE) tokenization instead of character-level
- Full-data, GPU-backed training run for more reliable, publication-grade numbers
- Push the FNet causal window size back up (32, 64+) now that the CPU cost is known, to see how
  performance scales with more context

---

*Built as a hands-on architecture comparison project — LSTM, GRU, and FNet implemented from scratch in
PyTorch with a shared, fair evaluation methodology.*
