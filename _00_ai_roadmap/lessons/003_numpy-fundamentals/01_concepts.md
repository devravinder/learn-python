# 01 — Concepts: NumPy Fundamentals

## Why not plain Python lists

A Python list stores pointers to arbitrary objects; a NumPy `ndarray` stores a
single fixed-size block of memory with one dtype. That means arithmetic runs as
tight compiled loops (C, sometimes SIMD-vectorized) instead of Python bytecode per
element — often 10–100x faster, and it's the reason every ML framework uses
array/tensor types instead of lists.

```python
import numpy as np

a = np.array([1, 2, 3, 4])
print(a.shape, a.dtype)   # (4,) int64
```

## Creating arrays

```python
np.zeros((2, 3))
np.ones((3,))
np.arange(0, 10, 2)          # [0,2,4,6,8]
np.linspace(0, 1, 5)         # 5 evenly spaced points from 0 to 1
np.random.default_rng(0).normal(size=(2, 2))   # reproducible random numbers
```

## Shape and reshaping

`shape` is a tuple of dimension sizes. A 2D array of shape `(3, 4)` has 3 rows,
4 columns. `-1` in `reshape` means "infer this dimension":

```python
a = np.arange(12)
b = a.reshape(3, 4)
c = a.reshape(3, -1)   # same as above, numpy infers 4
```

## Vectorized operations

Element-wise math applies to whole arrays at once — no explicit loop:

```python
x = np.array([1, 2, 3])
y = np.array([10, 20, 30])
x + y            # [11, 22, 33]
x * 2            # [2, 4, 6]
x ** 2           # [1, 4, 9]
np.dot(x, y)     # 1*10 + 2*20 + 3*30 = 140
```

## Broadcasting

NumPy lets you combine arrays of different shapes if their trailing dimensions are
compatible (equal, or one of them is 1):

```python
matrix = np.ones((3, 4))
row    = np.array([1, 2, 3, 4])
matrix + row     # row is broadcast across all 3 rows -> shape (3,4)

col = np.array([[1], [2], [3]])
matrix + col     # col is broadcast across all 4 columns -> shape (3,4)
```

This is exactly how "add a bias vector to every row of a batch" works in every
neural network layer.

## Indexing and slicing

```python
a = np.arange(20).reshape(4, 5)
a[0]          # first row
a[:, 0]       # first column
a[1:3, 2:4]   # sub-matrix (rows 1-2, cols 2-3)
a[a > 10]     # boolean mask -> 1D array of matching elements
```

## Aggregations and the `axis` argument

```python
a = np.arange(12).reshape(3, 4)
a.sum()          # sum over everything -> 66
a.sum(axis=0)    # sum down each column -> shape (4,)
a.sum(axis=1)    # sum across each row  -> shape (3,)
a.mean(axis=0)
```

Rule of thumb: `axis=0` collapses rows (result has one entry per column),
`axis=1` collapses columns (result has one entry per row).

## Matrix multiplication vs element-wise

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

A * B      # element-wise:      [[5,12],[21,32]]
A @ B      # matrix product:    [[19,22],[43,50]]
```

`@` (or `np.matmul`) is the operation behind every linear layer in a neural network
(`output = input @ weights + bias`).
