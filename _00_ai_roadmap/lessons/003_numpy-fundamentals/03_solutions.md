# 03 — Solutions: NumPy Fundamentals

## 1. Create and reshape

```python
import numpy as np

a = np.arange(30).reshape(5, 6)
```

## 2. Indexing

```python
a[2]          # 3rd row (index 2)
a[:, 1]       # 2nd column (index 1)
a[1:4, 2:5]   # rows 1-3, cols 2-4 (upper bound exclusive)
a[a > 15]     # flat array of matching elements
```

## 3. Element-wise ops and dot product

```python
a = np.arange(1, 6)          # [1,2,3,4,5]
b = np.arange(10, 60, 10)    # [10,20,30,40,50]

a + b   # [11,22,33,44,55]
a * b   # [10,40,90,160,250]
np.dot(a, b)   # 10+40+90+160+250 = 550

# manual check
total = sum(x * y for x, y in zip(a, b))
assert total == np.dot(a, b)
```

## 4. Broadcasting a bias vector

```python
ones = np.ones((4, 3))
bias = np.array([1, 2, 3])

result = ones + bias
# ones.shape = (4,3), bias.shape = (3,)
# trailing dimensions match (3 == 3), so bias is broadcast across all 4 rows
```

## 5. Random matrix stats

```python
rng = np.random.default_rng(42)
m = rng.integers(0, 10, size=(3, 4))

row_sums = m.sum(axis=1)
col_means = m.mean(axis=0)
overall_max = m.max()
overall_max_idx = np.unravel_index(np.argmax(m), m.shape)
```

`np.argmax` on a 2D array returns the index into the *flattened* array;
`np.unravel_index` converts that back to `(row, col)`.

## 6. Per-column min-max normalization

```python
def normalize_columns(x):
    col_min = x.min(axis=0)
    col_max = x.max(axis=0)
    return (x - col_min) / (col_max - col_min)

data = np.array([[1., 5., 10.], [2., 3., 20.], [3., 1., 30.]])
normalized = normalize_columns(data)
```

`axis=0` reduces over rows, producing one min/max per column; broadcasting then
applies each column's min/max to every row in that column automatically.

## 7. Batch @ weights

```python
A = np.random.default_rng(0).normal(size=(3, 4))
W = np.random.default_rng(1).normal(size=(4, 2))

out = A @ W
print(out.shape)   # (3, 2)
```

If `A` is 3 samples × 4 features and `W` is a linear layer mapping 4 input
features to 2 output features, `A @ W` produces the layer's output for all 3
samples at once — shape `(3, 2)`, i.e. 3 samples × 2 output units. This is
literally `nn.Linear(4, 2)` applied to a batch, minus the bias term.
