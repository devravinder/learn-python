# 02 — Practicals: K-Means Clustering

```python
from sklearn.datasets import make_blobs
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.8, random_state=0)
```

1. Implement K-Means from scratch (per `01_concepts.md`) with `k=4`. Plot
   the resulting clusters (colored by assigned label) with centroids marked.

2. Compare your from-scratch result to `sklearn.cluster.KMeans(n_clusters=4)`
   — cluster assignments may use different label *numbers* (cluster "0" in
   yours might be cluster "2" in sklearn's) but the actual groupings of
   points should match. Verify by checking that points grouped together in
   your version are also grouped together in sklearn's.

3. Run K-Means with 5 different random seeds for centroid initialization at
   `k=4`. Do you ever get a visibly different (worse) clustering, showing
   the local-minimum sensitivity from `01_concepts.md`? Report the inertia
   for each run.

4. Run the elbow method: fit K-Means for `k=1` through `k=10`, plot inertia
   vs `k`. Where's the elbow? Does it match the true `k=4` used to generate
   the data?

5. Compute silhouette score for `k=2` through `k=8`. Does the `k` that
   maximizes silhouette score agree with the elbow method's answer?

6. Generate a dataset where K-Means' spherical-cluster assumption clearly
   fails: `sklearn.datasets.make_moons(n_samples=300, noise=0.05)`. Fit
   `KMeans(n_clusters=2)` and plot the result — does it correctly separate
   the two crescents? Explain why or why not, referencing K-Means'
   assumptions.
