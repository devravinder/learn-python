# 01 — Concepts: Hierarchical Clustering

## Agglomerative (bottom-up) clustering

1. Start with every point as its own cluster.
2. Repeatedly merge the two **closest** clusters into one.
3. Continue until everything is in a single cluster.

This produces a full hierarchy of nested clusters — you decide *afterward*
how many clusters you want by "cutting" the hierarchy at some level, rather
than committing to `k` upfront like K-Means (Lesson 032).

## Linkage: what "closest clusters" means

Since a cluster (after the first merge) has multiple points, "distance
between clusters" needs a definition:

- **Single linkage**: distance between the closest pair of points (one from
  each cluster) — can produce long, straggly ("chained") clusters.
- **Complete linkage**: distance between the farthest pair — tends toward
  compact, evenly-sized clusters.
- **Average linkage**: average distance between all cross-cluster pairs —
  a middle ground.
- **Ward's method**: merges the pair that minimizes the resulting increase
  in within-cluster variance — often the best default, conceptually similar
  to what K-Means optimizes (inertia).

```python
from sklearn.cluster import AgglomerativeClustering
model = AgglomerativeClustering(n_clusters=3, linkage="ward")
labels = model.fit_predict(X)
```

## The dendrogram

A tree diagram showing every merge, with the y-axis representing the
distance at which clusters merged. Reading a dendrogram: the **height** of a
merge shows how (dis)similar the merged clusters were — long vertical lines
before a merge indicate genuinely distinct clusters being joined; cutting
horizontally at a given height gives you the clusters at that granularity.

```python
from scipy.cluster.hierarchy import dendrogram, linkage

Z = linkage(X, method="ward")
dendrogram(Z)
```

Choosing the number of clusters from a dendrogram: look for the **tallest
vertical gap** you can cut through without crossing a merge line — that
represents the point where continuing to merge would combine clusters that
are meaningfully different from each other.

## Agglomerative vs K-Means: tradeoffs

| | K-Means | Hierarchical (Agglomerative) |
|---|---|---|
| Need to specify `k` upfront | Yes | No (choose after seeing the dendrogram) |
| Deterministic | No (depends on init) | Yes (given a linkage method) |
| Scalability | Good (`O(n*k*iterations)`) | Poor (`O(n^2)` or `O(n^3)` depending on linkage — impractical for very large `n`) |
| Cluster shape assumption | Spherical | Depends on linkage; more flexible with single/average |
| Interpretability | Just final clusters | Full hierarchy — useful when relationships between clusters matter (e.g. taxonomy) |

## Divisive (top-down) clustering — brief mention

The reverse of agglomerative: start with everything in one cluster, split
recursively. Much less common in practice due to higher computational cost
per split decision; agglomerative is the standard approach when hierarchical
clustering is the tool of choice.

## When hierarchical clustering earns its cost

Best suited to smaller datasets where you want to understand *relationships
between clusters*, not just a final flat partition — e.g. grouping species by
similarity where sub-groupings are themselves meaningful, or exploratory
analysis where you're not sure how many clusters make sense and want to see
the full structure before deciding.
