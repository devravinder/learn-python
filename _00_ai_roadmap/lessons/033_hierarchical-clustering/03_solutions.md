# 03 — Solutions: Hierarchical Clustering

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering, KMeans
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

X, y_true = make_blobs(n_samples=60, centers=3, cluster_std=0.7, random_state=0)
```

## 1. Dendrogram

```python
Z = linkage(X, method="ward")
dendrogram(Z)
plt.show()
```

The tallest vertical gap (before the final merge into one cluster) should
sit above 3 clear branches, matching the true `centers=3` used to generate
the data.

## 2. Linkage comparison

```python
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, linkage_method in zip(axes, ["ward", "single", "complete"]):
    labels = AgglomerativeClustering(n_clusters=3, linkage=linkage_method).fit_predict(X)
    ax.scatter(X[:, 0], X[:, 1], c=labels, cmap="tab10")
    ax.set_title(linkage_method)
plt.show()
```

On well-separated blobs, all three linkage methods typically agree closely —
the differences between linkage methods matter much more on messier,
less-cleanly-separated, or non-spherical data (Q3).

## 3. Single linkage on elongated clusters

```python
rng = np.random.default_rng(0)
line1 = np.column_stack([np.linspace(0, 5, 30), rng.normal(0, 0.15, 30)])
line2 = np.column_stack([np.linspace(0, 5, 30), rng.normal(2, 0.15, 30)])
X_lines = np.vstack([line1, line2])

kmeans_labels = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(X_lines)
single_labels = AgglomerativeClustering(n_clusters=2, linkage="single").fit_predict(X_lines)

fig, axes = plt.subplots(1, 2, figsize=(9, 4))
axes[0].scatter(X_lines[:, 0], X_lines[:, 1], c=kmeans_labels, cmap="coolwarm")
axes[0].set_title("KMeans")
axes[1].scatter(X_lines[:, 0], X_lines[:, 1], c=single_labels, cmap="coolwarm")
axes[1].set_title("Single linkage")
plt.show()
```

K-Means, biased toward spherical/compact clusters, often splits each line
*vertically* into left/right halves rather than separating the two parallel
lines; single linkage's "closest point" merging criterion is well-suited to
following an elongated shape and typically separates the two lines
correctly — this exact tradeoff is why linkage choice matters, and why
"chaining" (usually described as single linkage's weakness) is actually a
strength for elongated cluster shapes specifically.

## 4. Cutting the dendrogram

```python
cut_height = 8   # pick based on your Q1 dendrogram's tallest gap
cluster_labels = fcluster(Z, t=cut_height, criterion="distance")
print(len(set(cluster_labels)))   # should be 3, matching the visual dendrogram read
```

## 5. Runtime comparison

```python
import time

for n in [100, 500, 2000]:
    Xn, _ = make_blobs(n_samples=n, centers=3, random_state=0)

    t0 = time.time()
    AgglomerativeClustering(n_clusters=3).fit(Xn)
    agg_time = time.time() - t0

    t0 = time.time()
    KMeans(n_clusters=3, n_init=10).fit(Xn)
    km_time = time.time() - t0

    print(n, "agglomerative:", agg_time, "kmeans:", km_time)
```

Agglomerative clustering's runtime should grow noticeably faster than
K-Means's as `n` increases — directly confirming the `O(n^2)`-or-worse vs
`O(n*k*iterations)` complexity difference from `01_concepts.md`, and why
hierarchical clustering is rarely used on very large datasets in practice.
