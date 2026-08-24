# 03 — Solutions: Eigenvalues, Eigenvectors & SVD

## 1. Eigen check

```python
import numpy as np

A = np.array([[2, 0], [0, 3]])
eigenvalues, eigenvectors = np.linalg.eig(A)

for i in range(len(eigenvalues)):
    v = eigenvectors[:, i]
    lam = eigenvalues[i]
    print(np.allclose(A @ v, lam * v))   # True for both
```

## 2. Orthogonal eigenvectors of a symmetric matrix

```python
S = np.array([[4, 2], [2, 3]])
vals, vecs = np.linalg.eig(S)
v1, v2 = vecs[:, 0], vecs[:, 1]
print(np.dot(v1, v2))   # ~0.0
```

## 3. PCA intuition on correlated data

```python
rng = np.random.default_rng(0)
x = rng.normal(0, 2, 500)
y = x * 0.8 + rng.normal(0, 0.5, 500)
data = np.column_stack([x, y])

centered = data - data.mean(axis=0)
cov = np.cov(centered.T)
vals, vecs = np.linalg.eig(cov)

top_direction = vecs[:, np.argmax(vals)]
print(top_direction)
```

Since `y` is mostly `0.8 * x` plus small noise, the data's main spread runs
diagonally along that relationship — the top eigenvector should point in
roughly that `[1, 0.8]`-ish direction (normalized), matching the visible
diagonal spread if you scatter-plot `data`.

## 4. SVD reconstruction

```python
rng = np.random.default_rng(1)
A = rng.normal(size=(5, 3))

U, S, Vt = np.linalg.svd(A, full_matrices=False)
reconstructed = U @ np.diag(S) @ Vt
print(np.allclose(A, reconstructed))   # True
```

`full_matrices=False` returns the "economy" SVD with compatible shapes
(`U`: 5x3, `S`: 3, `Vt`: 3x3) so the matrix product reconstructs `A` directly
without extra padding.

## 5. Low-rank approximation error

```python
rng = np.random.default_rng(2)
A = rng.normal(size=(6, 6))
U, S, Vt = np.linalg.svd(A)

def rank_k_approx(k):
    return U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]

err_2 = np.linalg.norm(A - rank_k_approx(2))
err_4 = np.linalg.norm(A - rank_k_approx(4))
print(err_2, err_4)   # err_4 < err_2
```

Keeping more singular values/vectors always gives an equal-or-better
(never worse) approximation, since each additional component can only add
more of the original matrix's structure back.

## 6. Why LoRA makes sense

A full weight matrix update during fine-tuning is often not truly "full
rank" in how much it actually needs to change — most of the useful update
lives in a lower-dimensional subspace. The Eckart–Young result says the best
rank-`k` approximation of any matrix is obtained via its top-`k` SVD
components, i.e. large matrices are frequently well-approximated by much
smaller ones. LoRA leans on that: instead of learning a full `ΔW` (as many
parameters as the original weight matrix), it learns `ΔW ≈ A @ B` with `A`
and `B` narrow — far fewer parameters, but able to capture most of the useful
update, exactly because real fine-tuning updates tend to be low-rank in
practice.
