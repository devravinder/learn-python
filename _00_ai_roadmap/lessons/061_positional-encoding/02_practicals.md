# 02 — Practicals: Positional Encoding

## Sinusoidal encoding (pure Python)

1. Implement `sinusoidal_encoding(seq_len, d_model)` per `01_concepts.md`'s
   formula. Compute it for `seq_len=6, d_model=8` and print the table.
   Confirm position 0's row is `[0, 1, 0, 1, 0, 1, 0, 1]` (since
   `sin(0)=0`, `cos(0)=1` for every dimension when `pos=0`).

2. Plot the encoding as a heatmap (`seq_len` on one axis, `d_model` on the
   other) for `seq_len=50, d_model=64`. Do low dimensions (left columns)
   visibly oscillate faster across positions than high dimensions (right
   columns), matching `01_concepts.md`'s "different frequencies per
   dimension" claim?

3. Compute cosine similarity (Lesson 010) between the encoding vectors for
   positions 0 and 1, positions 0 and 10, and positions 0 and 40 (using
   your Q1/Q2 table). Does similarity generally decrease as positions get
   farther apart — a sensible property for a *positional* signal to have?

## RoPE's relative-position property (pure Python, 2D simplification)

4. Implement a 2D rotation function `rotate(vec, theta)` (Lesson 011's
   rotation matrix, applied by hand to a 2D vector). For a fixed random
   query vector `q` and key vector `k`, compute
   `dot(rotate(q, pos_q), rotate(k, pos_k))` for three different `(pos_q,
   pos_k)` pairs that all have the **same difference** `pos_q - pos_k = 2`
   (e.g. `(3,1)`, `(5,3)`, `(10,8)`). Confirm the dot product is (almost)
   identical across all three pairs, despite the absolute positions being
   completely different.

5. Repeat Q4 with pairs having *different* differences (e.g. diff=2,
   diff=5, diff=8). Confirm the dot products now differ meaningfully
   across pairs — the rotated dot product depends on *relative* distance,
   not absolute position, exactly as `01_concepts.md` claims.

6. Explain in your own words why the property demonstrated in Q4-Q5 is
   useful for a language model specifically: what does it mean, in
   practical terms, for an attention score between two tokens to depend
   only on how far apart they are, rather than on their absolute position
   in the sequence?
