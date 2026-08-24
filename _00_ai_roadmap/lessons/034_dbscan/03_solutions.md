# 03 — Solutions: DBSCAN

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors

X, _ = make_moons(n_samples=300, noise=0.07, random_state=0)
```

## 1. DBSCAN on moons

```python
labels = DBSCAN(eps=0.2, min_samples=5).fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="coolwarm")
plt.show()
```

DBSCAN should correctly separate the two crescents into distinct clusters —
succeeding exactly where K-Means (Lesson 032) failed, since it doesn't
assume convex/spherical cluster shapes.

## 2. Injected noise points

```python
rng = np.random.default_rng(0)
noise_pts = rng.uniform(X.min(), X.max(), size=(15, 2))
X_with_noise = np.vstack([X, noise_pts])

labels2 = DBSCAN(eps=0.2, min_samples=5).fit_predict(X_with_noise)
n_noise = np.sum(labels2 == -1)
print(f"{n_noise} points labeled noise out of {len(labels2)}")
```

Most of the 15 injected scattered points should get labeled `-1` (noise),
since they're isolated and unlikely to have 5 neighbors within `eps=0.2` —
though a few might land close enough to the crescents or to each other by
chance to be included in a cluster or form their own tiny one.

## 3. Eps sweep

```python
for eps in [0.05, 0.1, 0.2, 0.3, 0.5]:
    labels = DBSCAN(eps=eps, min_samples=5).fit_predict(X_with_noise)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = np.sum(labels == -1)
    print(eps, "clusters:", n_clusters, "noise:", n_noise)
```

Very small `eps` (e.g. 0.05) typically produces many tiny clusters and a lot
of noise (not enough points fall within such a small radius); very large
`eps` (e.g. 0.5) tends to merge everything, including the injected noise
points, into a single cluster — the classic under/over-clustering pattern
that motivates careful `eps` selection via the k-distance plot.

## 4. K-distance plot

```python
neighbors = NearestNeighbors(n_neighbors=5).fit(X_with_noise)
distances, _ = neighbors.kneighbors(X_with_noise)
k_distances = np.sort(distances[:, -1])

plt.plot(k_distances)
plt.ylabel("distance to 5th nearest neighbor")
plt.show()
```

The "knee" — where the curve bends sharply upward — should sit somewhere in
the range that performed well in Q3 (often near `eps=0.15-0.2` for this
kind of data), confirming the k-distance heuristic gives a reasonable
starting point rather than needing pure trial and error.

## 5. Three clustering methods side by side

```python
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
db_labels = DBSCAN(eps=0.2, min_samples=5).fit_predict(X)
km_labels = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(X)
agg_labels = AgglomerativeClustering(n_clusters=2).fit_predict(X)

for ax, labels, title in zip(axes, [db_labels, km_labels, agg_labels], ["DBSCAN", "KMeans", "Agglomerative (ward)"]):
    ax.scatter(X[:, 0], X[:, 1], c=labels, cmap="coolwarm")
    ax.set_title(title)
plt.show()
```

DBSCAN should correctly separate the crescents; KMeans and
Agglomerative-with-Ward-linkage (which, like K-Means, favors compact
clusters) typically both fail to respect the crescent shapes, instead
splitting the data along a roughly straight boundary — visually confirming
which algorithms actually handle non-convex structure.

## 6. Varying density limitation

```python
rng = np.random.default_rng(1)
tight_cluster = rng.normal([0, 0], 0.2, size=(100, 2))
spread_cluster = rng.normal([6, 6], 1.5, size=(100, 2))
X_varying = np.vstack([tight_cluster, spread_cluster])

for eps in [0.3, 0.6, 1.0, 1.5]:
    labels = DBSCAN(eps=eps, min_samples=5).fit_predict(X_varying)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(eps, n_clusters, "noise:", np.sum(labels == -1))
```

Typically no single `eps` correctly identifies both clusters as exactly 2
dense groups: an `eps` small enough to avoid treating the sparse cluster's
points as isolated noise is often too small to properly connect the tight
cluster (fragmenting it or leaving edge points as noise), while an `eps`
large enough to fully connect the sparse cluster risks merging both
clusters together if they're not far enough apart. This directly confirms
DBSCAN's single-density-parameter limitation on genuinely
different-density clusters, motivating density-adaptive alternatives like
HDBSCAN for this specific scenario.
