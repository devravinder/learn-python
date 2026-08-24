# 02 — Practicals: NumPy Fundamentals

1. Create a 1D array of the integers 0–29, then reshape it to `(5, 6)`.

2. From that `(5, 6)` array, extract:
   - the 3rd row
   - the 2nd column
   - the sub-array of rows 1–3 and columns 2–4
   - all elements greater than 15 (as a flat array)

3. Create two arrays `a = np.arange(1, 6)` and `b = np.arange(10, 60, 10)`.
   Compute their element-wise sum, product, and dot product. Confirm the dot
   product manually (by hand or a plain Python loop) matches `np.dot`.

4. Create a `(4, 3)` array of ones and a 1D array `bias = np.array([1, 2, 3])`.
   Use broadcasting to add `bias` to every row. Explain in a comment why the
   shapes are compatible.

5. Create a `(3, 4)` matrix of random integers (use `np.random.default_rng(42)`
   for reproducibility) between 0 and 10. Compute:
   - the sum of each row
   - the mean of each column
   - the overall max and its index (`np.argmax`)

6. Implement min-max normalization of a 2D array **per column** (each column
   rescaled independently to `[0, 1]`) using only vectorized NumPy operations —
   no Python-level loops.

7. Given `A = np.random.default_rng(0).normal(size=(3, 4))` and
   `W = np.random.default_rng(1).normal(size=(4, 2))`, compute `A @ W` and
   report the resulting shape. Explain what this operation would represent if
   `A` were a batch of 3 samples with 4 features each, and `W` were a linear
   layer's weights.
