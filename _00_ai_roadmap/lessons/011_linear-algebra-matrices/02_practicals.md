# 02 — Practicals: Matrices & Operations

1. Given `A = np.array([[1, 2], [3, 4]])` and `B = np.array([[0, 1], [1, 0]])`,
   compute `A @ B` and `B @ A` by hand, then verify with NumPy. Confirm
   they're different — matrix multiplication doesn't commute.

2. Simulate a tiny "linear layer": `X` is a batch of 5 samples with 4
   features (`X = rng.normal(size=(5, 4))`), `W` is a weight matrix mapping 4
   features to 3 outputs (`W = rng.normal(size=(3, 4))`), `b` is a bias
   `(3,)`. Compute `output = X @ W.T + b` and report its shape. Explain in a
   comment why `W.T` (not `W`) is needed for the shapes to work.

3. Compute the determinant and rank of:
   ```python
   singular = np.array([[1, 2], [2, 4]])   # second row is 2x the first
   ```
   Confirm the determinant is 0 and the rank is 1 (not 2). Explain
   geometrically what this matrix does to the unit square.

4. Solve the linear system `2x + y = 5`, `x - y = 1` using
   `np.linalg.solve` (set up `A` and `b` yourself). Verify your answer by
   substituting back in.

5. Given a `(4, 3)` matrix `X`, compute the Gram matrix `X.T @ X`. Confirm it
   is symmetric (`np.allclose(gram, gram.T)`).

6. Generate a random orthogonal 2D rotation matrix for angle `θ = 45°`:
   ```
   R = [[cos θ, -sin θ], [sin θ, cos θ]]
   ```
   Apply it to the vector `[1, 0]` and confirm the result's length is
   unchanged (`np.linalg.norm`). Confirm `R.T @ R` is (approximately) the
   identity matrix.
