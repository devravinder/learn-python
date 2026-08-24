# 02 — Practicals: Polynomial Regression & Regularization

Reuse Lesson 017's cubic dataset:

```python
import numpy as np
rng = np.random.default_rng(0)
x = rng.uniform(-3, 3, 60).reshape(-1, 1)
y = 0.5 * x.ravel()**3 - 2 * x.ravel() + rng.normal(0, 3, 60)
```

1. Fit a degree-15 polynomial with plain `LinearRegression` (via
   `PolynomialFeatures` + a pipeline) and report test MSE (80/20 split).

2. Fit the same degree-15 polynomial features with `Ridge(alpha=1.0)`
   instead, and compare test MSE to Q1. Does regularization reduce
   overfitting here?

3. Sweep `alpha` over `[0.001, 0.01, 0.1, 1, 10, 100]` for Ridge on the
   degree-15 features; plot test MSE vs alpha (log x-axis). Find the best
   alpha empirically.

4. Generate a dataset with 20 features where only 3 are actually predictive
   (the rest are pure noise):
   ```python
   rng = np.random.default_rng(1)
   X = rng.normal(size=(200, 20))
   true_weights = np.zeros(20)
   true_weights[[2, 7, 15]] = [5, -3, 2]
   y = X @ true_weights + rng.normal(0, 1, 200)
   ```
   Fit `Lasso(alpha=0.5)` and print the learned coefficients. How many are
   exactly (or nearly) zero? Do the nonzero ones line up with indices 2, 7,
   15?

5. Fit `Ridge(alpha=0.5)` on the same data from Q4 and compare its
   coefficients to Lasso's — confirm Ridge shrinks the irrelevant features'
   coefficients toward zero but doesn't zero them out exactly.

6. Standardize the features from Q4 first, then refit both Ridge and Lasso —
   does the result change noticeably compared to unstandardized features
   (note: `rng.normal` already produces roughly-standardized data, so try
   this on Project 002's `housing.csv` features instead, where scales differ
   wildly, to see the effect clearly).
