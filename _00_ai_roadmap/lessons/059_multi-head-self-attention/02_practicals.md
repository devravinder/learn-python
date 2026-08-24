# 02 — Practicals: Multi-Head Self-Attention

## Causal masking (pure Python, extending Lesson 058)

1. Implement a version of `softmax_row` that treats `-inf` inputs as
   contributing exactly `0` after softmax (careful with `max()` over a row
   containing `-inf` — take the max only over finite values).

2. Build a `(5,5)` causal mask (position `i` can attend to `j <= i` only,
   Lesson 059's lower-triangular pattern) and apply it to a random `(5,5)`
   score matrix by setting masked positions to `-inf` before your masked
   softmax. Confirm: row 0's weights are `[1,0,0,0,0]` (can only attend to
   itself), and every row still sums to exactly 1 over its *allowed*
   positions.

3. Confirm the causal mask genuinely prevents "seeing the future": change
   a score in a masked-out (upper-triangular) position to a huge value
   (e.g. `1000`) and confirm it has **zero** effect on any row's output
   weights — the mask must be applied *before* softmax, not after.

## PyTorch multi-head attention

4. Implement `MultiHeadAttention` from `01_concepts.md`. Create a random
   input `(batch=2, seq_len=6, d_model=32)` and run it through with
   `n_heads=4`. Confirm the output shape matches the input shape
   (`(2, 6, 32)`) — multi-head attention is shape-preserving, which is
   exactly what lets you stack many Transformer blocks (Lesson 060).

5. Add causal masking to your `MultiHeadAttention` (per `01_concepts.md`'s
   `masked_fill`). Verify by checking that changing a *later* token in the
   input sequence doesn't change an *earlier* token's output (a direct,
   practical test of the causal property — compute the output twice with
   different values at position 5, and confirm positions 0-4's outputs are
   identical both times).

6. Compare parameter counts: single-head attention at `d_model=32`
   (`d_k=32`) vs 4-head attention at `d_model=32` (`d_k=8` per head).
   Confirm the total parameter count in the `W_q`/`W_k`/`W_v`/`W_o`
   matrices is the **same** either way — multi-head attention doesn't add
   parameters over single-head at the same `d_model`, it just restructures
   how they're used (per `01_concepts.md`'s efficiency point).
