# 02 — Practicals: Build a GPT, Part 1

Use any plain text you have on hand (a few paragraphs is enough to test
the mechanics; a real training run in Lesson 065 will want much more).

## Data pipeline (can be done in pure Python first, then PyTorch)

1. Build the character-level vocabulary, `encode`/`decode` functions, and
   confirm `decode(encode(text)) == text` exactly (a lossless round-trip
   is a hard requirement for any tokenizer).

2. Split into train (90%) and validation (10%) sets **by position** (not
   randomly shuffled — this is one long continuous text, and shuffling
   would leak future context into "earlier" training positions in a way
   that doesn't reflect how the model will actually be used).

3. Implement `get_batch` (per `01_concepts.md`) using **plain Python lists**
   first (no PyTorch) — sample `batch_size` random starting positions,
   extract `block_size`-length chunks for `x`, and the same chunks shifted
   by one position for `y`. Print one `(x, y)` pair and confirm
   `y[i] == x[i+1]` for all but the last position (Lesson 063's shift-by-one,
   verified directly on real sampled data).

4. Convert your pure-Python `get_batch` to use PyTorch tensors
   (`torch.randint`, `torch.stack`). Confirm it produces the same shapes as
   the pure-Python version for the same `batch_size`/`block_size`.

## Assembling and shape-checking the model

5. Instantiate Lesson 060's `GPT` class with your real vocabulary size and
   a small config (`d_model=64, n_heads=4, n_layers=4, d_ff=256`). Run one
   batch from `get_batch` through it and confirm the output shape is
   `(batch_size, block_size, vocab_size)`.

6. Compute the cross-entropy loss (Lesson 063) on this single untrained
   batch. Given `vocab_size` unique characters, what loss would you expect
   from a **completely untrained** (random-weight) model, in theory (hint:
   relate to Lesson 063's "uniform random model" perplexity result)? Does
   your model's actual initial loss come out close to that expectation?
