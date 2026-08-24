# 01 — Concepts: Curse of Dimensionality

## The core problem: volume grows exponentially with dimensions

To cover 10% of the range along each axis of a `d`-dimensional cube, you need
a fraction `0.1^d` of the total volume — for `d=1` that's 10% of the data,
for `d=10` it's `0.1^10 = 0.0000000001`% . To keep the same *data density* as
dimensions grow, you'd need exponentially more data — in practice, you never
have enough, so your data becomes increasingly **sparse** in high-dimensional
space.

## Distances become less meaningful

As dimensionality grows, the ratio between the distance to the *nearest*
point and the distance to the *farthest* point tends toward 1 — everything
becomes roughly equidistant. This directly breaks distance-based algorithms:

- **KNN** (Lesson 025): "nearest neighbors" stops being a meaningful concept
  when all points are nearly the same distance away.
- **K-Means** (Lesson 032): cluster assignments based on distance become
  unreliable for the same reason.

```python
import numpy as np

def distance_ratio(n_points, dims):
    rng = np.random.default_rng(0)
    points = rng.uniform(0, 1, size=(n_points, dims))
    query = rng.uniform(0, 1, size=dims)
    dists = np.linalg.norm(points - query, axis=1)
    return dists.min() / dists.max()

for d in [2, 10, 100, 1000]:
    print(d, distance_ratio(1000, d))
# ratio approaches 1.0 as dims grows -> nearest and farthest look similar
```

## Overfitting risk grows with dimensions

More features relative to your number of samples means more ways for a model
to find spurious patterns that fit training data by chance (directly
connects to Lesson 017's variance problem) — this is why "just add every
feature you can think of" tends to backfire without enough data or
regularization to compensate.

## Not purely a curse: high dimensions can help too

Real high-dimensional data (images, text embeddings) typically lies on or
near a much lower-dimensional **manifold** embedded in the high-dimensional
space — e.g. natural images are a tiny fraction of all possible pixel
combinations. This is why dimensionality reduction (PCA, Lesson 031;
autoencoders) works at all: the "true" dimensionality of the underlying
pattern is often far lower than the raw feature count suggests.

## Practical mitigations

- **Feature selection**: drop irrelevant/redundant features before training.
- **Dimensionality reduction** (PCA, Lesson 031): project onto the directions
  that actually carry variance/information.
- **Regularization** (Lesson 022): constrain the model so it can't exploit
  spurious high-dimensional patterns as easily.
- **More data**: the direct (if often impractical) fix — density scales
  with both sample count and dimensionality.
- **Domain-appropriate architectures**: CNNs (Lesson 043) exploit the
  structure of image data (nearby pixels are related) instead of treating
  every pixel as an independent, unstructured dimension — a big part of why
  they outperform naive fully-connected networks on images.

## Why this matters heading into embeddings and LLMs

Word/token embeddings (Lesson 050+) deliberately map a huge, sparse
vocabulary (tens of thousands of possible tokens, one-hot = extremely
high-dimensional and empty) into a much smaller, dense space (a few hundred
to a few thousand dimensions) where distances and directions are actually
meaningful — a direct, practical answer to the curse of dimensionality for
language data.
