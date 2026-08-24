# 02 — Practicals: Principal Component Analysis

```python
from sklearn.datasets import load_digits
digits = load_digits()
X, y = digits.data, digits.target   # 1797 images, 64 pixels each (8x8), digit labels 0-9
```

1. Implement PCA from scratch (per `01_concepts.md`) and reduce `X` to 2
   components. Plot the result as a scatter plot colored by `y` (the digit
   label — note PCA never sees `y`, it's unsupervised; you're only using
   labels afterward to check if the unsupervised structure happens to align
   with them).

2. Compare your from-scratch result to `sklearn.decomposition.PCA(n_components=2)`
   — confirm the projected points match (up to a possible sign flip per
   component, which is mathematically valid — eigenvectors are only defined
   up to sign).

3. Fit `PCA()` with no `n_components` limit (keeps all components) on the
   standardized digits data. Plot the cumulative explained variance ratio
   against number of components. How many components are needed to explain
   90% of the variance?

4. Reduce the digits data to that many components (Q3), then train a
   `LogisticRegression` classifier on the reduced features vs the original
   64 features. Compare test accuracy and training time — is the
   dimensionality reduction worth it here?

5. Take one digit image (a single 64-dim vector), project it to 2
   components and then reconstruct it back to 64 dimensions
   (`inverse_transform`). Display the original vs reconstructed image
   (`.reshape(8, 8)` + `plt.imshow`) — how much detail is lost keeping only
   2 components? Repeat with the number of components from Q3 and compare
   visually.

6. Generate a dataset with 2 genuinely informative dimensions plus 50 pure
   noise dimensions (similar to Lesson 019's setup). Apply PCA to reduce
   back to 2 components, then run KNN classification before and after PCA —
   confirm PCA recovers most of the accuracy lost to the curse of
   dimensionality in Lesson 019.
