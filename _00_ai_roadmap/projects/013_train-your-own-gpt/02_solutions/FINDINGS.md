# Findings — Train Your Own GPT

*(The tokenizer training and sizing numbers below are verified — actually
run against the real generated `fallback_corpus.txt` in this sandbox
(pure Python, no PyTorch needed for the tokenizer). The actual GPT
training run itself was **not** executed here — no PyTorch available in
the authoring sandbox — so training-loss/generated-text results are
described as expected behavior. Run `train.py` yourself and replace this
section with your real loss curves and samples.)*

## Tokenizer training (verified)

Training the from-scratch `BPETokenizer` (Lesson 068a) on the 660,370-byte
synthetic fallback corpus with `vocab_size=512` (256 merges):

- **Training time: ~26 seconds** (pure Python, no optimization beyond
  what Lesson 068a's implementation already does — real production BPE
  trainers use more efficient data structures for large corpora, but this
  is fine at hobby scale).
- **Compression: 660,370 bytes → 134,855 tokens (4.90x)** — a strong ratio,
  substantially helped by this fallback corpus's high repetitiveness
  (templated sentences), which is not representative of real prose.

## Sizing decision (verified arithmetic, Lesson 068's ratio)

`train_data` after the 90/10 split is ~121,370 tokens. The Chinchilla-
style ratio (`N ≈ D / 20`) suggests a compute-optimal model size of only
**~6,070 parameters** for this corpus size.

**A real, honest tension worth confronting directly**: even a deliberately
tiny GPT config (`d_model=32, n_layers=2, n_heads=2, d_ff=128`) comes out
to roughly **~59,000 parameters** — about **10x more** than the
Chinchilla-suggested size for this corpus. This is a genuine, useful
finding, not a failure to fix: it means **this particular fallback corpus
is too small and too repetitive to justify even a "small" GPT
configuration** — exactly the scenario Lesson 068 warned about, now faced
directly rather than abstractly. Two honest responses: (1) accept the
mismatch for a first pipeline-correctness run (this is what the reference
`train.py` defaults do), since the point is verifying the mechanism works,
not achieving compute-optimal training; or (2) use a real, larger,
less-repetitive text corpus (a real public-domain book, likely 5-50x
larger and far richer), which is exactly what the project brief recommends
for genuinely interesting generated text.

## Expected training behavior

With the default config on the fallback corpus, expect training loss to
drop quickly and substantially (the corpus's heavy repetition makes it
"easy" in an information-theoretic sense — Lesson 016's entropy framing:
low-entropy data is easier to model) — likely reaching a low perplexity
within a modest number of steps, precisely *because* the data is
repetitive rather than because the model has learned rich language
structure. Validation loss should track training loss reasonably closely
given the same repeated templates appear throughout both splits.

## Expected generated text quality

On the repetitive fallback corpus specifically, expect generated text to
closely mimic the templated sentence structure (`"[subject] [verb] [object],
[ending]."`) with plausible but limited variety — a direct reflection of
what the training data actually contained. **This is not a fair test of
the architecture's real capability** — Lesson 063a's simple bigram/MLP
models would likely do reasonably well on data this repetitive too. The
real, meaningful comparison (a noticeably more coherent, more varied,
longer-range-consistent output than a bigram model could produce) requires
training on genuinely rich text — a real book's worth of natural prose —
which is exactly why the project brief recommends sourcing real text for
your actual submission rather than relying on the fallback corpus alone.
