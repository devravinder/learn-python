# 03 — Solutions: Curse of Dimensionality

## 1. Distance ratio vs dimension

```python
import numpy as np
import matplotlib.pyplot as plt

def distance_ratio(n_points, dims, seed=0):
    rng = np.random.default_rng(seed)
    points = rng.uniform(0, 1, size=(n_points, dims))
    query = rng.uniform(0, 1, size=dims)
    dists = np.linalg.norm(points - query, axis=1)
    return dists.min() / dists.max()

dims = [1, 2, 5, 10, 50, 100, 500]
ratios = [distance_ratio(1000, d) for d in dims]

plt.plot(dims, ratios, marker="o")
plt.xscale("log")
plt.xlabel("dimensions")
plt.ylabel("min/max distance ratio")
plt.show()

for d, r in zip(dims, ratios):
    print(d, r)
```

The ratio typically climbs well above 0.9 somewhere in the 50–100 dimension
range for this setup — beyond that point, "nearest neighbor" and "farthest
point" are barely distinguishable by distance alone.

## 2. Volume concentrating away from the center

```python
def fraction_in_sphere(d, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    points = rng.uniform(-0.5, 0.5, size=(n, d))   # cube centered at origin
    dists = np.linalg.norm(points, axis=1)
    return (dists < 0.5).mean()

for d in [2, 10, 50]:
    print(d, fraction_in_sphere(d))
```

The fraction inside the inscribed sphere shrinks sharply as `d` grows
(from a sizeable fraction at `d=2` down to a tiny fraction at `d=50`) —
volume increasingly concentrates in the "corners" of the cube, away from the
center, as dimensionality increases.

## 3–4. KNN degrading with noise dimensions, PCA recovering it

```python
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

rng = np.random.default_rng(0)
n = 200
X2d = np.vstack([
    rng.normal([0, 0], 1, size=(n // 2, 2)),
    rng.normal([4, 4], 1, size=(n // 2, 2)),
])
y = np.array([0] * (n // 2) + [1] * (n // 2))

knn = KNeighborsClassifier(n_neighbors=5)
score_2d = cross_val_score(knn, X2d, y, cv=5).mean()

noise = rng.normal(0, 1, size=(n, 200))
X_noisy = np.hstack([X2d, noise])
score_noisy = cross_val_score(knn, X_noisy, y, cv=5).mean()

X_pca = PCA(n_components=2).fit_transform(X_noisy)
score_pca = cross_val_score(knn, X_pca, y, cv=5).mean()

print("2D only:", score_2d)
print("2D + 200 noise dims:", score_noisy)
print("PCA back to 2D:", score_pca)
```

Accuracy on the noisy high-dimensional version typically drops substantially
toward chance level, because the 200 irrelevant dimensions dominate the
Euclidean distance calculation and drown out the 2 genuinely informative
ones (consistent with Q1's distance-concentration effect). PCA, by finding
the directions of maximum variance, should recover most of the informative
structure and bring accuracy back close to the original 2D-only score —
since the true signal *is* the top 2 principal components in this
construction.

## 5. Why "more features is free" is false

More features means: (a) each individual point's Euclidean neighborhood
becomes increasingly meaningless as noise dimensions swamp the signal in
distance calculations (Q1–Q2, Q3), and (b) with a fixed amount of training
data, more features gives a model more opportunities to fit spurious
correlations between noise features and the label by chance — the same
overfitting/variance problem from Lesson 017, made worse by every added,
uninformative dimension. Both effects mean added features only help if
they carry real signal *and* you have enough data to estimate their effect
reliably — otherwise they actively hurt.
