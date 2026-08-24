# 01 — Concepts: Vectors & Vector Spaces

## What a vector is, in ML terms

Geometrically: a point in space, or an arrow from the origin. In ML: an
ordered list of numbers representing anything — a data sample's features
`[age, income, height]`, a word's embedding `[0.2, -1.4, 0.7, ...]`, an
image's flattened pixels, a model's entire weight vector.

```python
import numpy as np
v = np.array([3, 4])
```

## Magnitude (norm) and direction

**L2 norm** (Euclidean length): `||v|| = sqrt(sum(v_i^2))`. For `[3, 4]`,
`||v|| = 5`.

**L1 norm** (Manhattan length): `||v||_1 = sum(|v_i|)`. Used in Lasso
regularization (Lesson 022) because it encourages exact zeros in a way L2
doesn't.

```python
l2 = np.linalg.norm(v)          # 5.0
l1 = np.linalg.norm(v, ord=1)   # 7.0
```

**Unit vector**: `v / ||v||` — same direction, length 1. Normalizing
embeddings to unit length before comparing them is standard practice.

## Vector operations

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

a + b       # element-wise addition: [5, 7, 9]
a - b       # [-3, -3, -3]
2 * a       # scalar multiplication: [2, 4, 6]
```

**Linear combination**: `c1*v1 + c2*v2 + ...` — scaling and adding vectors.
Almost every operation in ML (a weighted sum of features, an ensemble
average, an attention-weighted sum of value vectors) is a linear combination.

## Dot product

```
a · b = sum(a_i * b_i) = ||a|| ||b|| cos(θ)
```

```python
np.dot(a, b)   # or a @ b
```

Two equivalent readings: (1) algebraic — multiply corresponding entries and
sum; (2) geometric — related to the angle `θ` between the vectors. This dual
meaning is why the dot product shows up everywhere: it's simultaneously "how
aligned are these two vectors" and "a cheap thing to compute."

## Cosine similarity

```
cos(θ) = (a · b) / (||a|| ||b||)
```

Ranges from -1 (opposite direction) to 1 (same direction), 0 = orthogonal
(unrelated). This is *the* standard way to compare embeddings (word vectors,
sentence embeddings, image embeddings) — it measures direction similarity
while ignoring magnitude, which matters because embedding magnitude often
just reflects frequency/confidence, not meaning.

```python
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
```

## Orthogonality and projection

Two vectors are **orthogonal** if `a · b = 0` (90° apart) — geometrically
independent directions. The **projection** of `a` onto `b` is the component
of `a` pointing in `b`'s direction:

```
proj_b(a) = (a · b / ||b||^2) * b
```

Projection is the geometric operation behind linear regression (projecting
the target vector onto the column space of the features) and PCA (projecting
data onto the directions of maximum variance).

## Vector spaces and basis (just enough to be dangerous)

A **vector space** is a set of vectors closed under addition and scalar
multiplication. A **basis** is a minimal set of vectors that can combine
(via linear combinations) to reach every point in the space; the number of
basis vectors is the space's **dimension**. In ML, "dimensionality" of your
data almost always means "how many numbers describe one sample" — i.e. which
vector space your data vectors live in.
