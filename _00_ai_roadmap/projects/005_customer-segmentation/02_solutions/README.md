# Reference Solution

```bash
python data/generate_data.py
python analysis.py
```

- [analysis.py](analysis.py) — elbow/silhouette k-selection, PCA
  visualization, KMeans + Agglomerative + DBSCAN cross-check, segment
  profiling
- [FINDINGS.md](FINDINGS.md) — verified segment profiles, an honest note on
  cluster-recovery instability, and business recommendations per segment

Try [01_requirement.md](../01_requirement.md) yourself first. If your K=4
result looks different from `FINDINGS.md`'s (e.g. it cleanly finds 4 evenly
distinct segments instead of splitting the top spenders), that's not
necessarily wrong — compare silhouette scores and note that K-Means'
sensitivity to initialization is itself one of this project's lessons.
