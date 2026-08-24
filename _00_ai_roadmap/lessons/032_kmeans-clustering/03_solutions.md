# 03 — Solutions: K-Means Clustering

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.8, random_state=0)
```

## 1. From-scratch K-Means

```python
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

labels, centroids = kmeans(X, 4)
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="tab10")
plt.scatter(centroids[:, 0], centroids[:, 1], c="black", marker="x", s=200)
plt.show()
```

## 2. Compare to sklearn

```python
sklearn_model = KMeans(n_clusters=4, n_init=10, random_state=0).fit(X)
sklearn_labels = sklearn_model.labels_

# check grouping equivalence rather than exact label match
from sklearn.metrics import adjusted_rand_score
print(adjusted_rand_score(labels, sklearn_labels))   # should be close to 1.0
```

Adjusted Rand Index close to 1.0 confirms the two implementations found the
same groupings, even if the arbitrary cluster-number assignments differ.

## 3. Sensitivity to initialization

```python
for seed in range(5):
    lbls, cents = kmeans(X, 4, seed=seed)
    inertia = sum(np.sum((X[lbls == i] - cents[i])**2) for i in range(4))
    print(seed, inertia)
```

With well-separated blobs like this, results are often stable across seeds;
try `cluster_std=2.5` (more overlap) to make bad-initialization sensitivity
more visible — some seeds may converge to a clearly worse (higher inertia)
local minimum with overlapping clusters.

## 4. Elbow method

```python
inertias = []
for k in range(1, 11):
    m = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
    inertias.append(m.inertia_)

plt.plot(range(1, 11), inertias, marker="o")
plt.xlabel("k")
plt.ylabel("inertia")
plt.show()
```

The elbow should appear around `k=4`, matching the true number of generating
centers — a case where the elbow method works cleanly, which isn't always
true on messier real data.

## 5. Silhouette score

```python
scores = []
for k in range(2, 9):
    m = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
    scores.append(silhouette_score(X, m.labels_))

plt.plot(range(2, 9), scores, marker="o")
plt.show()
print("best k by silhouette:", list(range(2, 9))[np.argmax(scores)])
```

Should also point to `k=4` on this clean dataset, agreeing with the elbow
method — a good sign both diagnostics are reliable here specifically because
the clusters are genuinely well-separated and spherical.

## 6. K-Means failing on non-spherical clusters

```python
X_moons, _ = make_moons(n_samples=300, noise=0.05, random_state=0)
moon_labels = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(X_moons)

plt.scatter(X_moons[:, 0], X_moons[:, 1], c=moon_labels, cmap="coolwarm")
plt.show()
```

K-Means will cut the two crescents roughly in half with a straight-ish
boundary instead of separating them by crescent — it has no way to represent
a non-convex cluster shape, since it only ever assigns points to their
*nearest centroid*, which fundamentally produces convex (specifically,
Voronoi-cell-shaped) cluster regions. This is the direct motivation for
density-based clustering (DBSCAN, Lesson 034), which doesn't assume any
particular cluster shape.
