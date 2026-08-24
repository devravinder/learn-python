# 03 — Solutions: Vectors & Vector Spaces

## 1. Norms

```python
import numpy as np

a = np.array([3, -4, 12])
l1 = np.abs(a).sum()                 # 19
l2 = np.linalg.norm(a)               # 13.0
unit = a / l2
print(np.linalg.norm(unit))          # 1.0
```

## 2. Cosine similarity

```python
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

king  = np.array([0.9, 0.1, 0.2])
queen = np.array([0.85, 0.15, 0.25])
apple = np.array([0.1, 0.9, 0.05])

print(cosine_similarity(king, queen))  # close to 1, e.g. ~0.995
print(cosine_similarity(king, apple))  # much lower, e.g. ~0.28
```

## 3. Projection and orthogonality

```python
a = np.array([2., 0.])
b = np.array([1., 1.])

proj = (np.dot(a, b) / np.dot(b, b)) * b   # [1, 1]
residual = a - proj                          # [1, -1]
print(np.dot(residual, b))                   # 0.0 -> orthogonal
```

## 4. Most similar users

```python
users = np.array([
    [5, 4, 1],
    [4, 5, 1],
    [1, 1, 5],
    [2, 1, 4],
])

best_pair, best_sim = None, -2
for i in range(len(users)):
    for j in range(i + 1, len(users)):
        sim = cosine_similarity(users[i], users[j])
        if sim > best_sim:
            best_sim, best_pair = sim, (i, j)

print(best_pair, best_sim)   # users 0 and 1 (both like action-ish movies 1&2, dislike 3)
```

## 5. Why cosine over Euclidean for embeddings

If you scale an embedding vector by 10x (same direction, 10x magnitude),
cosine similarity with any other vector is **unchanged** (the formula
divides out both norms), but Euclidean distance changes a lot. Embedding
magnitude in many models reflects incidental factors (word frequency,
training dynamics) rather than meaning, so a similarity measure that ignores
magnitude and focuses purely on direction is more robust for "are these two
things semantically similar."

## 6. Vectorized row-wise L2 normalization

```python
def normalize_rows(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / norms

m = np.array([[3., 4.], [1., 1.], [5., 0.]])
normalized = normalize_rows(m)
print(np.linalg.norm(normalized, axis=1))   # [1., 1., 1.]
```

`axis=1` computes each row's norm independently; `keepdims=True` keeps the
result shape `(n, 1)` instead of `(n,)` so broadcasting divides each row by
its own norm rather than failing on a shape mismatch.
