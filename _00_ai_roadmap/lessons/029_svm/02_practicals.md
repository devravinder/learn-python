# 02 — Practicals: Support Vector Machines

```python
from sklearn.datasets import make_blobs, make_circles
X_lin, y_lin = make_blobs(n_samples=200, centers=2, cluster_std=1.5, random_state=0)
X_circ, y_circ = make_circles(n_samples=200, noise=0.1, factor=0.4, random_state=0)
```

1. Fit `SVC(kernel="linear")` on `X_lin, y_lin`. Plot the data and the
   support vectors (`model.support_vectors_`) highlighted differently from
   other points.

2. Sweep `C` over `[0.01, 0.1, 1, 10, 100]` on `X_lin, y_lin` (with some
   overlap added: use `cluster_std=3.0` instead so classes aren't cleanly
   separable). Report accuracy and the number of support vectors at each
   `C`. Does a larger `C` use fewer or more support vectors, and why (relate
   to margin width)?

3. Fit `SVC(kernel="linear")` on `X_circ, y_circ` (the circles data — not
   linearly separable) and report accuracy. Then fit
   `SVC(kernel="rbf")` and compare. Confirm the kernel trick recovers good
   accuracy where the linear kernel fails.

4. Sweep `gamma` over `[0.1, 1, 10, 100]` for the RBF kernel on `X_circ,
   y_circ`. Does very high `gamma` show signs of overfitting (check train
   vs test accuracy, not just train accuracy)?

5. Compare `SVC(kernel="rbf")` to `KNeighborsClassifier` (Lesson 025) on the
   circles dataset — both are capable of nonlinear boundaries via different
   mechanisms (kernel trick vs local neighborhoods). Which performs better
   here, and does one train/predict faster?

6. Standardize features (mean 0, std 1) before fitting an RBF SVM on a
   dataset where one feature has a much larger raw scale than another
   (reuse a variant of Lesson 025 Q4's setup). Confirm unscaled features
   hurt SVM in the same way they hurt KNN, and explain why (both rely on
   distance/dot-product calculations).
