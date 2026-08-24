# 02 — Practicals: K-Nearest Neighbors

```python
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=300, noise=0.3, random_state=0)
```

1. Split into train/test. Fit `KNeighborsClassifier` with `k=1`, `k=15`, and
   `k=100`. Report test accuracy for each — which is likely overfit,
   underfit, or reasonable, and why (relate to `01_concepts.md`'s bias-
   variance framing)?

2. Sweep `k` from 1 to 50 and plot train accuracy and test accuracy against
   `k`. Identify the best `k` by test accuracy.

3. Implement KNN classification completely from scratch (no `sklearn`):
   given a query point, compute Euclidean distance to all training points,
   take the `k` closest, and return the majority label. Verify it matches
   `sklearn`'s predictions for `k=5` on a few test points.

4. Create two features on very different scales (`feature_a` range 0-1,
   `feature_b` range 0-10000) where only `feature_a` is actually predictive.
   Fit KNN without scaling, then with `StandardScaler` applied first.
   Compare test accuracy — quantify how much unscaled features hurt KNN
   here.

5. Compare `metric="euclidean"` vs `metric="manhattan"` on the `make_moons`
   data at the best `k` from Q2 — does the choice of distance metric matter
   much for this dataset?

6. Time (`time.time()`) predicting on 100 test points with training set
   sizes of 500, 5000, and 20000 (generate larger synthetic datasets with
   `make_moons`). Confirm prediction time grows with training set size,
   consistent with KNN's `O(n)`-per-prediction cost.
