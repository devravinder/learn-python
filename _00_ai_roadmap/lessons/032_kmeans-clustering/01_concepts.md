# 01 — Concepts: K-Means Clustering

## The algorithm

1. Pick `k` (number of clusters). Initialize `k` **centroids** (cluster
   centers) — commonly random data points, or better, via **k-means++**
   (spreads initial centroids apart, reducing bad-initialization risk).
2. **Assign step**: assign every point to its nearest centroid (Euclidean
   distance, Lesson 010).
3. **Update step**: move each centroid to the mean of the points assigned
   to it.
4. Repeat steps 2-3 until assignments stop changing (converged).

```python
import numpy as np

def kmeans(X, k, n_iters=100, seed=0):
    rng = np.random.default_rng(seed)
    centroids = X[rng.choice(len(X), k, replace=False)]
    for _ in range(n_iters):
        distances = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
        labels = np.argmin(distances, axis=1)
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        if np.allclose(new_centroids, centroids):
            break
        centroids = new_centroids
    return labels, centroids
```

`X[:, None, :] - centroids[None, :, :]` broadcasts (Lesson 003) every point
against every centroid at once, computing all pairwise distances without an
explicit loop.

## What K-Means is optimizing

Minimizes **inertia** (within-cluster sum of squared distances to centroid):

```
inertia = Σ_clusters Σ_points_in_cluster ||point - centroid||^2
```

This is a non-convex optimization (Lesson 015's convexity discussion) — the
algorithm above (Lloyd's algorithm) is guaranteed to decrease inertia every
iteration and converge, but only to a **local** minimum, which depends on
initial centroid placement. Standard fix: run K-Means multiple times with
different random initializations, keep the result with lowest inertia
(`sklearn`'s `n_init` parameter does this automatically).

```python
from sklearn.cluster import KMeans
model = KMeans(n_clusters=3, n_init=10, random_state=0)
labels = model.fit_predict(X)
```

## Choosing `k`: the elbow method

Plot inertia against `k` for a range of values — inertia always decreases as
`k` increases (more clusters can always fit the data at least as well), but
usually shows a bend ("elbow") where additional clusters stop helping much.
Pick `k` near the elbow — a judgment call, not an exact formula.

```python
inertias = []
for k in range(1, 11):
    model = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
    inertias.append(model.inertia_)
```

## Silhouette score — a second opinion on `k`

Measures how well-separated clusters are: for each point, compares its
average distance to points in its own cluster vs the nearest other cluster.
Ranges from -1 (likely wrong cluster) to 1 (well-separated); doesn't rely on
eyeballing an elbow, and can be compared numerically across different `k`
values directly.

```python
from sklearn.metrics import silhouette_score
score = silhouette_score(X, labels)
```

## Assumptions and limitations

K-Means assumes clusters are **roughly spherical and similar in size** —
it struggles with elongated, differently-sized, or non-convex cluster shapes
(this is exactly the situation DBSCAN, Lesson 034, handles better). It's also
sensitive to feature scale (Euclidean distance again — standardize first,
Lesson 008) and to outliers (a single far-away point can pull a centroid
noticeably, since the mean isn't robust to outliers — Lesson 008 again).

## K-Means is unsupervised — there's no "accuracy"

Since there's no ground truth label to compare against (typically), you
can't compute accuracy the way Module 4 did. Inertia and silhouette score
measure internal cluster quality; if you *do* have external labels
available for validation purposes only (not used during clustering), metrics
like Adjusted Rand Index compare cluster assignments to those labels — but
the whole point of clustering is usually that you don't have labels to begin
with.
