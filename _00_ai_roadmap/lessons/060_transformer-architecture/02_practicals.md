# 02 — Practicals: The Transformer Architecture

## LayerNorm from scratch (pure Python)

1. Implement `layer_norm(x, gamma=1.0, beta=0.0, eps=1e-5)` per
   `01_concepts.md`'s formula, operating on a single feature vector (a list
   of numbers). Apply it to `[2, 4, 4, 4, 5, 5, 7, 9]` and confirm the
   result has mean ≈ 0 and variance ≈ 1.

2. Explain, using a concrete example, why LayerNorm (normalize across
   features, per position) is more suitable than BatchNorm (normalize
   across the batch, per feature) when a batch contains sequences of
   different lengths padded with `[PAD]` tokens (Lesson 055) — what would
   padding tokens do to a BatchNorm's per-feature batch statistics that
   LayerNorm avoids entirely?

## PyTorch: assembling the full block

3. Implement `FeedForward` and `TransformerBlock` from `01_concepts.md`
   (reuse your `MultiHeadAttention` from Lesson 059). Run a random
   `(batch=2, seq_len=8, d_model=32)` input through one block with
   `n_heads=4, d_ff=128` and confirm the output shape matches the input.

4. Stack 4 `TransformerBlock`s and run the same input through all of them
   in sequence. Confirm the final output shape is unchanged — this
   shape-preservation property is exactly what lets you stack an arbitrary
   number of blocks.

5. Assemble the full `GPT` class from `01_concepts.md` with a small
   vocabulary (`vocab_size=50`), `d_model=32`, `n_heads=4`, `n_layers=2`,
   `d_ff=128`, `max_len=20`. Run a random integer sequence
   `torch.randint(0, 50, (1, 10))` through it and confirm the output shape
   is `(1, 10, 50)` — one probability distribution over the vocabulary per
   input position.

6. Count total trainable parameters in your small GPT
   (`sum(p.numel() for p in model.parameters())`). Then look up (or
   estimate from published architecture details) GPT-2 small's parameter
   count (~124M) and note how many orders of magnitude larger it is than
   your toy model — a concrete sense of scale before Module 11 has you
   train a real (if still small) version yourself.
