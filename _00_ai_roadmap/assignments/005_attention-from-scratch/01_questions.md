# 01 — Questions

Work in plain Python (lists of lists for matrices) for Q1-Q5 — no NumPy,
no PyTorch. This is intentionally more tedious than using a library; that
tedium is exactly what makes the verification in Q6 meaningful.

1. Implement matrix helpers: `matmul`, `transpose`, `add` (elementwise),
   and `linear(x, W, b)` (computes `x @ W + b`, i.e. a full `nn.Linear`
   forward pass by hand).

2. Implement `softmax_row` (numerically stable, Lesson 007's max-subtraction
   trick) and `scaled_dot_product_attention(Q, K, V, mask=None)` — where
   `mask` is an optional matrix of 0s/1s; masked positions (`mask[i][j]==0`)
   should get an effective score of `-inf` before softmax.

3. Implement `multi_head_attention(x, W_q, b_q, W_k, b_k, W_v, b_v, W_o, b_o, n_heads)`:
   project `x` to `Q`, `K`, `V` via `linear`, **split each into `n_heads`
   equal-width chunks** along the feature dimension, run
   `scaled_dot_product_attention` independently per head (optionally with
   a shared causal mask), **concatenate** the per-head outputs back
   together, then apply the output projection `linear(concat, W_o, b_o)`.

4. Build a causal mask for a sequence length of your choice (lower-
   triangular 0/1 matrix, Lesson 059) and run your `multi_head_attention`
   on a small random input (e.g. 5 positions, 8 features, 2 heads) with
   that mask applied. Confirm position 0's output only actually depends on
   position 0's input (test this the same way Lesson 059's practicals
   did: change a later position's input, confirm earlier outputs are
   unaffected).

5. Run the same input **without** a mask and compare to Q4's masked
   version, position by position. Before running it, predict which
   position(s) should come out **identical** between the masked and
   unmasked versions, and which should differ — think about which
   position's causal-mask row already includes every other position
   anyway. Then check your prediction against the actual output.

6. **Verification**: build the exact same weights (`W_q`, `b_q`, etc.) as
   PyTorch tensors, implement the equivalent forward pass with
   `torch.nn.functional` operations (or reuse Lesson 059's
   `MultiHeadAttention`, feeding in the same weights manually via
   `state_dict` or direct assignment), and confirm your pure-Python
   output matches PyTorch's output to within `1e-4` per element.
