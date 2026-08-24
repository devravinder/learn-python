# 01 — Concepts: Principal Component Analysis

## Supervised vs unsupervised — a mental reset

Every model since Lesson 020 learned from `(X, y)` pairs — a target to
predict. Starting here, Module 5 has **no `y`** — the goal is to find
structure in `X` alone: compress it (PCA), group it (K-Means, Lesson 032), or
find its shape (DBSCAN, Lesson 034). This isn't "worse" than supervised
learning, just a different question: not "predict this label" but "what does
this data look like."

## What PCA does

Finds a new set of axes (**principal components**) — linear combinations of
the original features — ordered so the first captures the most variance in
the data, the second the most *remaining* variance (orthogonal to the
first), and so on. Keeping only the top few components compresses the data
while retaining as much information (variance) as possible.

## The algorithm (you already have all the pieces from Lesson 012)

1. **Center** the data (subtract each feature's mean) — required so
   variance is measured around the data's own center, not the origin.
2. Compute the **covariance matrix** of the centered data.
3. Find its **eigenvectors and eigenvalues** (Lesson 012) — eigenvectors are
   the principal component directions, eigenvalues are how much variance
   each one explains.
4. Sort by eigenvalue descending; keep the top `k`.
5. **Project** the centered data onto the top `k` eigenvectors: `X_reduced =
   X_centered @ top_k_eigenvectors`.

```python
import numpy as np

def pca(X, k):
    X_centered = X - X.mean(axis=0)
    cov = np.cov(X_centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)   # eigh: for symmetric matrices, sorted ascending
    order = np.argsort(eigenvalues)[::-1]
    top_k = eigenvectors[:, order[:k]]
    return X_centered @ top_k, eigenvalues[order]
```

(`np.linalg.eigh` instead of `eig`: covariance matrices are always symmetric
— Lesson 011 — so `eigh` is both faster and numerically more stable.)

## Explained variance ratio

`eigenvalue_i / sum(all eigenvalues)` tells you what fraction of the data's
total variance each component captures. Plotting cumulative explained
variance against number of components (a **scree plot**) is the standard way
to choose `k`: pick enough components to retain e.g. 90-95% of variance.

```python
from sklearn.decomposition import PCA
model = PCA(n_components=0.95)   # keep enough components for 95% variance
X_reduced = model.fit_transform(X)
print(model.explained_variance_ratio_)
```

## What PCA is and isn't good for

**Good for**: visualization (project to 2-3 dimensions to plot
high-dimensional data), noise reduction (later components often capture
mostly noise), fighting the curse of dimensionality before KNN/clustering
(Lesson 019), speeding up training on very high-dimensional data, removing
multicollinearity (Lesson 021 — principal components are, by construction,
uncorrelated with each other).

**Not good for**: preserving interpretability (a principal component is a
weighted mix of original features, not one meaningful thing on its own);
capturing *non-linear* structure (PCA only finds linear combinations — for
that, look at t-SNE/UMAP, or autoencoders once you reach neural networks);
guaranteeing the top components are relevant to a *specific downstream task*
(PCA optimizes for variance, which might not align with what's predictive
for your particular label, if you have one).

## Standardization before PCA

PCA is scale-sensitive (Lesson 008) — a feature with a naturally larger
numeric range will dominate the variance calculation and thus the
components, regardless of its actual importance. **Standardize features
before PCA** unless you have a specific reason not to (e.g. all features
already share meaningful, comparable units).
