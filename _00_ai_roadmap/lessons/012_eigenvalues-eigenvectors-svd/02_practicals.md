# 02 — Practicals: Eigenvalues, Eigenvectors & SVD

1. For `A = np.array([[2, 0], [0, 3]])`, compute eigenvalues/eigenvectors
   with `np.linalg.eig`. Confirm `A @ v ≈ λ * v` for each eigenvector/
   eigenvalue pair returned.

2. For a symmetric matrix `S = np.array([[4, 2], [2, 3]])`, compute its
   eigenvectors and confirm they are orthogonal to each other (dot product
   ≈ 0).

3. Generate a synthetic correlated dataset:
   ```python
   rng = np.random.default_rng(0)
   x = rng.normal(0, 2, 500)
   y = x * 0.8 + rng.normal(0, 0.5, 500)
   data = np.column_stack([x, y])
   ```
   Center the data (subtract the mean), compute its covariance matrix
   (`np.cov(data.T)`), and find its eigenvectors. Which eigenvector has the
   larger eigenvalue, and does its direction visually match the main spread
   of a scatter plot of the data?

4. Compute the SVD of a random `(5, 3)` matrix `A`. Verify
   `U @ np.diag(S) @ Vt` reconstructs `A` (up to floating point error) — note
   you'll need to pad `S` into the right diagonal shape, or use
   `full_matrices=False`.

5. Take a `(6, 6)` random matrix, compute its SVD, and reconstruct a rank-2
   approximation using only the top 2 singular values/vectors. Compute the
   Frobenius norm of the difference between the original and the
   approximation (`np.linalg.norm(A - A_approx)`) and compare it to using
   rank-4 instead — confirm more retained components means lower error.

6. Explain, in your own words, why LoRA (learn a low-rank update `A @ B`
   instead of a full weight matrix) is a reasonable thing to try, given what
   you now know about low-rank approximation and the Eckart–Young intuition.
