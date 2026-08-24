# 01 — Concepts: Eigenvalues, Eigenvectors & SVD

## Eigenvectors and eigenvalues

For a square matrix `A`, a vector `v` is an **eigenvector** if applying `A`
only scales it, without changing its direction:

```
A @ v = λ * v
```

`λ` (a scalar) is the corresponding **eigenvalue**. Geometrically: most
vectors get rotated *and* scaled by `A`; eigenvectors are the special
directions that only get scaled.

```python
import numpy as np
A = np.array([[2, 0], [0, 3]])
eigenvalues, eigenvectors = np.linalg.eig(A)
```

For a **symmetric** matrix (like a covariance matrix), eigenvectors are
always orthogonal to each other — this is exactly why PCA's principal
components (the eigenvectors of the covariance matrix) are perpendicular
directions.

## Why eigenvectors matter: PCA intuition

The covariance matrix of a dataset describes how features vary together. Its
eigenvector with the **largest eigenvalue** points in the direction of
maximum variance in the data — the single direction along which the data is
most "spread out." Projecting data onto the top few eigenvectors (Lesson 031)
compresses it to fewer dimensions while keeping as much information
(variance) as possible.

## Singular Value Decomposition (SVD)

Eigendecomposition only works cleanly for square matrices. SVD generalizes
the idea to **any** `(m, n)` matrix:

```
A = U @ Σ @ V.T
```

- `U` (`m x m`, orthogonal): output-space directions
- `Σ` (`m x n`, diagonal): **singular values** — non-negative, sorted
  largest to smallest, analogous to "how much stretching" happens along each
  direction
- `V.T` (`n x n`, orthogonal): input-space directions

```python
U, S, Vt = np.linalg.svd(A)
```

## SVD and low-rank approximation

Keeping only the top `k` singular values/vectors gives the **best possible
rank-`k` approximation** of `A` (in a precise mathematical sense — this is
the Eckart–Young theorem, though you don't need to prove it, just use it):

```python
k = 2
A_approx = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
```

This is the exact mathematical idea behind:
- **PCA** — SVD of the (centered) data matrix directly gives the principal
  components, without ever explicitly forming the covariance matrix.
- **LoRA** (Lesson 070) — instead of fine-tuning a full weight matrix `W`
  (huge), learn a low-rank update `ΔW = A @ B` where `A`, `B` are small — the
  same "a big matrix can often be well-approximated by a low-rank one"
  intuition, applied to make fine-tuning cheap.
- **Recommender systems** — decomposing a user-item ratings matrix into
  low-rank user/item factor matrices.

## Positive semi-definite matrices (a preview for optimization)

A symmetric matrix `A` is **positive semi-definite (PSD)** if all its
eigenvalues are `>= 0` (equivalently, `x.T @ A @ x >= 0` for every vector
`x`). This matters in Lesson 015+: the Hessian (matrix of second
derivatives) of a convex loss function is PSD everywhere, which is precisely
what guarantees gradient descent won't get stuck in a bad local shape.
