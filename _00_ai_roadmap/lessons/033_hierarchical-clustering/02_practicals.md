# 02 — Practicals: Hierarchical Clustering

```python
from sklearn.datasets import make_blobs
X, y_true = make_blobs(n_samples=60, centers=3, cluster_std=0.7, random_state=0)
```

1. Compute `scipy.cluster.hierarchy.linkage(X, method="ward")` and plot the
   dendrogram. How many clusters does the tallest gap suggest?

2. Fit `AgglomerativeClustering(n_clusters=3)` with `linkage="ward"`,
   `"single"`, and `"complete"`. Plot all three results side by side
   (colored scatter plots). Do they agree on this well-separated data?

3. Generate an elongated, non-spherical cluster shape (e.g. two parallel
   lines of points with some noise) where K-Means (Lesson 032) would
   struggle. Compare `AgglomerativeClustering(linkage="single")` against
   `KMeans` on this data — does single linkage's "chaining" behavior help or
   hurt here?

4. Using the dendrogram from Q1, manually pick a cut height and use
   `scipy.cluster.hierarchy.fcluster` to extract cluster labels at that
   height. Confirm the resulting number of clusters matches your visual
   read of the dendrogram.

5. Compare runtime (`time.time()`) of `AgglomerativeClustering` vs `KMeans`
   as the dataset grows: try `n_samples = [100, 500, 2000]`. Confirm
   hierarchical clustering's runtime grows much faster, consistent with its
   worse time complexity from `01_concepts.md`.
