# 02 — Practicals: DBSCAN

```python
from sklearn.datasets import make_moons
X, _ = make_moons(n_samples=300, noise=0.07, random_state=0)
```

1. Fit `DBSCAN(eps=0.2, min_samples=5)` on the moons data. Plot the result
   (noise points, label `-1`, in a distinct color). Does it correctly
   separate the two crescents, unlike K-Means in Lesson 032?

2. Add 15 random uniform noise points scattered across the plot's range to
   `X`. Refit DBSCAN with the same parameters — are most of the injected
   noise points correctly labeled `-1`?

3. Sweep `eps` over `[0.05, 0.1, 0.2, 0.3, 0.5]` (fixed `min_samples=5`) on
   the noisy moons data from Q2. Report the number of clusters found and the
   number of noise points at each `eps`. What happens at very small `eps`
   (too many tiny clusters/noise) and very large `eps` (everything merges
   into one cluster)?

4. Compute and plot the k-distance plot (per `01_concepts.md`, `k=5`) for
   the noisy moons data. Identify the "knee" and check whether the `eps`
   value it suggests matches what worked well in Q3.

5. Compare `DBSCAN`, `KMeans(n_clusters=2)`, and
   `AgglomerativeClustering(n_clusters=2)` side by side on the noisy moons
   data (3 subplots). Which correctly identifies the two crescent shapes?

6. Generate two clusters with very different densities (one tight, one
   spread out) and try to find a single `eps`/`min_samples` that correctly
   separates both — report whether you succeed, and if not, explain why,
   connecting to the "varying density" limitation in `01_concepts.md`.
