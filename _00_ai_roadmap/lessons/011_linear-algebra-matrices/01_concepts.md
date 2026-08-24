# 01 — Concepts: Matrices & Operations

## A matrix is two things at once

1. **A batch of vectors** — a `(n, d)` matrix is `n` rows, each a `d`-dim
   vector (your entire dataset, or a batch fed to a model).
2. **A linear transformation** — a `(d_out, d_in)` matrix maps any `d_in`-dim
   vector to a `d_out`-dim vector via multiplication. A neural network layer
   *is* a matrix (plus a bias and nonlinearity).

## Matrix multiplication

```python
import numpy as np
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
A @ B
```

`(A @ B)[i, j] = sum_k A[i, k] * B[k, j]` — row `i` of `A` dotted with
column `j` of `B`. Shapes must match: `(m, n) @ (n, p) -> (m, p)`. This is
**not** commutative (`A @ B != B @ A` in general) — order matters, unlike
scalar multiplication.

## Transpose

```python
A.T   # swap rows and columns
```

Used constantly to make shapes line up (`X @ W.T` in a linear layer) and in
`A.T @ A` (the Gram matrix, central to PCA and least-squares regression).

## Identity and inverse

The **identity matrix** `I` leaves any vector unchanged: `I @ v = v`. The
**inverse** `A^-1` (when it exists) undoes `A`'s transformation:
`A^-1 @ A = I`. Solving `Ax = b` for `x` is `x = A^-1 @ b` — this is exactly
what "solve the normal equations" means in closed-form linear regression
(Lesson 020).

```python
A_inv = np.linalg.inv(A)
x = np.linalg.solve(A, b)   # preferred over computing A_inv explicitly - more numerically stable
```

Not every matrix has an inverse — only **square, full-rank** matrices do. A
matrix without an inverse is called **singular**.

## Determinant

A single number summarizing how a matrix scales area/volume:
`det(A) = 0` means `A` is singular (not invertible) — it collapses space
into a lower dimension (e.g. squishes a 2D plane onto a line).

```python
np.linalg.det(A)
```

## Rank

The number of linearly independent rows (or columns) of a matrix — the true
dimensionality of the space it can reach via linear combinations. A matrix is
**full rank** if `rank == min(rows, cols)`. Low rank relative to its size
means redundant information — the conceptual basis for PCA and low-rank
approximation (used in LoRA, Lesson 070, to fine-tune LLMs cheaply).

```python
np.linalg.matrix_rank(A)
```

## Special matrix types worth recognizing

- **Diagonal**: nonzero only on the diagonal — represents independent
  per-dimension scaling.
- **Symmetric**: `A == A.T` — covariance matrices and Gram matrices are
  always symmetric.
- **Orthogonal**: `A.T @ A == I` — represents a pure rotation/reflection,
  preserves lengths and angles. Rotation matrices in positional encodings
  (RoPE, Lesson 061) are orthogonal.

## Why this matters for neural networks specifically

A single linear layer computes `output = input @ W.T + b`, where `input` is
a `(batch, d_in)` matrix and `W` is `(d_out, d_in)`. Stacking layers is
stacking matrix multiplications (with nonlinearities between them, or the
whole network would collapse into one big linear transformation — see why in
Lesson 036). Attention (Lesson 058) is *entirely* matrix multiplications:
`Q @ K.T` for similarity scores, then the result `@ V` for the weighted sum.
