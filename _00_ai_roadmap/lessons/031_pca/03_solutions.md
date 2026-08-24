# 03 — Solutions: Principal Component Analysis

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

digits = load_digits()
X, y = digits.data, digits.target
```

## 1. From-scratch PCA

```python
def pca_from_scratch(X, k):
    X_centered = X - X.mean(axis=0)
    cov = np.cov(X_centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = np.argsort(eigenvalues)[::-1]
    top_k = eigenvectors[:, order[:k]]
    return X_centered @ top_k, eigenvalues[order]

X_2d, eigvals = pca_from_scratch(X, 2)
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap="tab10", s=10)
plt.colorbar()
plt.show()
```

Digit clusters should visibly separate somewhat, even though PCA never saw
the labels — a sign the pixel-variance structure it found does correlate
with digit identity.

## 2. Compare to sklearn

```python
sklearn_pca = PCA(n_components=2).fit_transform(X - X.mean(axis=0))
print(np.allclose(np.abs(X_2d), np.abs(sklearn_pca), atol=1e-6))
```

Comparing absolute values (or checking correlation instead of exact
equality) accounts for the sign-flip ambiguity — both implementations should
otherwise match closely.

## 3. Scree plot / cumulative explained variance

```python
X_scaled = StandardScaler().fit_transform(X)
full_pca = PCA().fit(X_scaled)
cumulative = np.cumsum(full_pca.explained_variance_ratio_)

plt.plot(cumulative)
plt.axhline(0.9, color="red", linestyle="--")
plt.xlabel("number of components")
plt.ylabel("cumulative explained variance")
plt.show()

n_for_90 = np.argmax(cumulative >= 0.9) + 1
print(n_for_90)   # typically somewhere around 20-25 out of 64 for this dataset
```

## 4. Reduced vs full features for classification

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import time

X_reduced = PCA(n_components=n_for_90).fit_transform(X_scaled)
Xtr_full, Xte_full, ytr, yte = train_test_split(X_scaled, y, test_size=0.2, random_state=0)
Xtr_red, Xte_red, _, _ = train_test_split(X_reduced, y, test_size=0.2, random_state=0)

t0 = time.time()
LogisticRegression(max_iter=2000).fit(Xtr_full, ytr).score(Xte_full, yte)
print("full features time:", time.time() - t0)

t0 = time.time()
acc_reduced = LogisticRegression(max_iter=2000).fit(Xtr_red, ytr).score(Xte_red, yte)
print("reduced features time:", time.time() - t0, "accuracy:", acc_reduced)
```

Accuracy on ~20-25 PCA components is typically very close to using all 64
raw features, while training is faster on the smaller input — dimensionality
reduction paying off with little to no accuracy cost on this dataset.

## 5. Reconstruction quality

```python
sample = X[0].reshape(1, -1)

pca_2 = PCA(n_components=2).fit(X)
reconstructed_2 = pca_2.inverse_transform(pca_2.transform(sample))

pca_n = PCA(n_components=n_for_90).fit(X)
reconstructed_n = pca_n.inverse_transform(pca_n.transform(sample))

fig, axes = plt.subplots(1, 3, figsize=(9, 3))
axes[0].imshow(sample.reshape(8, 8), cmap="gray"); axes[0].set_title("original")
axes[1].imshow(reconstructed_2.reshape(8, 8), cmap="gray"); axes[1].set_title("2 components")
axes[2].imshow(reconstructed_n.reshape(8, 8), cmap="gray"); axes[2].set_title(f"{n_for_90} components")
plt.show()
```

The 2-component reconstruction should look like a blurry, barely-recognizable
smear; the `n_for_90`-component reconstruction should look close to the
original — a direct visual demonstration of the variance-vs-compression
tradeoff.

## 6. PCA recovering accuracy lost to noise dimensions

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

rng = np.random.default_rng(0)
n = 200
X2d = np.vstack([rng.normal([0, 0], 1, (n//2, 2)), rng.normal([4, 4], 1, (n//2, 2))])
y2 = np.array([0]*(n//2) + [1]*(n//2))
noise = rng.normal(0, 1, (n, 50))
X_noisy = np.hstack([X2d, noise])

knn = KNeighborsClassifier(n_neighbors=5)
print("noisy, no PCA:", cross_val_score(knn, X_noisy, y2, cv=5).mean())

X_pca = PCA(n_components=2).fit_transform(X_noisy)
print("noisy, with PCA:", cross_val_score(knn, X_pca, y2, cv=5).mean())
```

This should closely reproduce Lesson 019's finding: KNN accuracy on the raw
noisy features drops toward chance, while PCA-reduced features (which
recover the 2 genuinely informative directions as the top principal
components, since they carry far more variance than the noise) restore
accuracy close to the noise-free 2D baseline.
