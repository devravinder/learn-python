# 01 — Concepts: DBSCAN

## The core idea: clusters are dense regions

**DBSCAN** (Density-Based Spatial Clustering of Applications with Noise)
defines a cluster as a region of high point density, separated from other
clusters by regions of low density — a fundamentally different definition
than K-Means's "close to a centroid" or hierarchical clustering's "merged
via linkage."

## Two parameters, three point types

- **`eps`**: radius defining a point's neighborhood.
- **`min_samples`**: minimum number of points (including itself) required
  within `eps` for a point to count as dense.

Every point is classified as:
- **Core point**: has at least `min_samples` points within `eps` of it.
- **Border point**: not a core point itself, but within `eps` of a core
  point — gets assigned to that core point's cluster.
- **Noise point**: neither — not part of any cluster.

## The algorithm

1. For each unvisited point, check if it's a core point (enough neighbors
   within `eps`).
2. If it is, start a new cluster and recursively add all points
   density-reachable from it (directly or through a chain of core points).
3. Points that end up reachable from no core point are labeled **noise**
   (`-1` in `sklearn`).

```python
from sklearn.cluster import DBSCAN
model = DBSCAN(eps=0.5, min_samples=5)
labels = model.fit_predict(X)
noise_points = X[labels == -1]
```

## Why this handles arbitrary shapes and outliers

Since clusters grow by chaining together nearby dense regions rather than
assuming any particular geometric shape, DBSCAN naturally handles the
crescent-moon shapes that broke K-Means in Lesson 032, and any other
non-convex structure — as long as the cluster is genuinely a connected
dense region. Outlier detection falls out for free: sparse, isolated points
simply never become part of any dense chain, and get labeled noise
automatically — neither K-Means nor (without extra post-processing)
hierarchical clustering does this.

## Choosing `eps` and `min_samples`

- **`min_samples`**: a common rule of thumb is `2 * number_of_features`, but
  it's ultimately about how much "not-noise" evidence you want to require.
- **`eps`**: the **k-distance plot** helps — compute each point's distance
  to its `k`-th nearest neighbor (`k = min_samples`), sort ascending, plot.
  Look for the "knee" in the curve — points to the left of the knee are in
  dense regions (good `eps` choices are near the knee's y-value); points far
  to the right are increasingly isolated.

```python
from sklearn.neighbors import NearestNeighbors
neighbors = NearestNeighbors(n_neighbors=5).fit(X)
distances, _ = neighbors.kneighbors(X)
k_distances = np.sort(distances[:, -1])
plt.plot(k_distances)   # look for the knee
```

## Limitations

- **Struggles with varying density**: a single `eps`/`min_samples` pair
  assumes roughly uniform density across all real clusters — a dataset with
  one dense cluster and one sparse cluster may need different parameters for
  each, which DBSCAN can't do simultaneously (HDBSCAN, a hierarchical
  extension, addresses this — worth knowing exists, not covered in depth
  here).
- **High-dimensional data**: like KNN and K-Means, DBSCAN relies on distance,
  so the curse of dimensionality (Lesson 019) applies — consider PCA
  (Lesson 031) first for high-dimensional data.
- **Parameter sensitivity**: results can change noticeably with small
  `eps` adjustments, more so than K-Means's `k`.

## Choosing among K-Means, Hierarchical, and DBSCAN

| Situation | Best fit |
|---|---|
| Roughly spherical, similar-sized clusters, known/guessable `k` | K-Means |
| Want the full nested cluster structure, smaller dataset | Hierarchical |
| Arbitrary cluster shapes, need automatic outlier detection, don't know `k` | DBSCAN |
