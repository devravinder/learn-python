# 02 — Practicals: The Attention Mechanism

## Pure Python (no dependencies — small enough to hand-verify)

1. Implement `matmul`, `transpose`, and `softmax_row` from scratch using
   plain Python lists (no NumPy). Implement
   `scaled_dot_product_attention(Q, K, V)` per `01_concepts.md`'s formula.

2. Create small `Q`, `K`, `V` matrices (4 positions, 8-dim vectors, random
   values) and run your attention function. Confirm every row of the
   attention weight matrix sums to exactly 1 (it's a valid probability
   distribution per query position, Lesson 007).

3. **Demonstrate why scaling matters**: with `d_k=64` (a realistic head
   dimension) and `Q`, `K` drawn from a standard normal distribution,
   compute raw dot-product scores for one query against all keys. Compare
   the softmax of those raw scores to the softmax of the *scaled* scores
   (divided by `sqrt(64)`). Is the unscaled version visibly more
   "peaked" (closer to one-hot) than the scaled version?

4. Construct a toy scenario where attention should clearly favor one
   position: make `V` such that position 2's value vector is very
   distinctive (e.g. all `10.0`s, others near 0), and set `Q`/`K` so that
   query 0 has a high dot product specifically with key 2. Confirm the
   attention output for query 0 is dominated by value 2's vector.

## PyTorch

5. Implement the same `scaled_dot_product_attention` using PyTorch tensors
   and `torch.nn.functional.softmax`. Confirm it matches your from-scratch
   version's output on the same `Q`, `K`, `V` (convert your Python lists to
   tensors).

6. Visualize an attention weight matrix as a heatmap
   (`seaborn.heatmap` or `plt.imshow`) for a short sentence's self-attention
   (use random `Q`, `K`, `V` derived from random embeddings — you don't
   have a trained model yet, so the pattern will be meaningless, but the
   *mechanics* of producing and reading a `(seq_len, seq_len)` attention
   map are exactly what you'll do with trained models in Lesson 060+).
