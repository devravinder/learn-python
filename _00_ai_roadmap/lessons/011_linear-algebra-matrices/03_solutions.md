# 03 — Solutions: Matrices & Operations

## 1. Non-commutativity

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[0, 1], [1, 0]])

print(A @ B)   # [[2,1],[4,3]]
print(B @ A)   # [[3,4],[1,2]]
```

Different results confirm `A @ B != B @ A` in general.

## 2. Linear layer shapes

```python
rng = np.random.default_rng(0)
X = rng.normal(size=(5, 4))   # 5 samples, 4 features
W = rng.normal(size=(3, 4))   # maps 4 -> 3
b = rng.normal(size=(3,))

output = X @ W.T + b
print(output.shape)   # (5, 3)
```

`X` is `(5, 4)` and `W` is `(3, 4)`; `X @ W` would need `W` shaped `(4, k)`
to match, but `W` is `(3, 4)` — transposing to `W.T` gives `(4, 3)`, making
`X @ W.T` a valid `(5,4) @ (4,3) -> (5,3)` multiplication: 5 samples, each
now with 3 output values.

## 3. Singular matrix

```python
singular = np.array([[1, 2], [2, 4]])
print(np.linalg.det(singular))          # 0.0
print(np.linalg.matrix_rank(singular))  # 1
```

Since row 2 is exactly 2x row 1, the matrix maps every input onto a single
line (rank 1) instead of spanning the full 2D plane — geometrically, it
squashes the unit square down to a line segment with zero area, which is
exactly what a zero determinant means.

## 4. Solving a linear system

```python
A = np.array([[2, 1], [1, -1]])
b = np.array([5, 1])
x = np.linalg.solve(A, b)
print(x)   # [2, 1]  ->  x=2, y=1

# check: 2*2+1=5 ✓,  2-1=1 ✓
```

## 5. Gram matrix symmetry

```python
X = rng.normal(size=(4, 3))
gram = X.T @ X
print(np.allclose(gram, gram.T))   # True
```

`(X.T @ X).T = X.T @ X.T.T = X.T @ X` algebraically, which is why any Gram
matrix is always symmetric regardless of `X`.

## 6. Rotation matrix

```python
theta = np.radians(45)
R = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta),  np.cos(theta)],
])

v = np.array([1., 0.])
rotated = R @ v
print(np.linalg.norm(rotated), np.linalg.norm(v))   # both 1.0

print(np.allclose(R.T @ R, np.eye(2)))   # True -> R is orthogonal
```

A rotation preserves length by definition (`R` is orthogonal), which is
exactly why `R.T @ R == I`.
