# 02 — Practicals: Logistic Regression

```python
import numpy as np
rng = np.random.default_rng(0)
n = 300
hours_studied = rng.uniform(0, 10, n)
z = 1.5 * hours_studied - 7 + rng.normal(0, 1, n)   # linear combo + noise
passed = (z > 0).astype(int)
```

1. Fit `sklearn.linear_model.LogisticRegression` on `hours_studied` (reshape
   to 2D) predicting `passed`. Print the coefficient and intercept.

2. Implement sigmoid from scratch (Lesson 013) and manually compute
   `P(passed=1)` for `hours_studied = 5` using your fitted model's
   coefficient/intercept. Confirm it matches `model.predict_proba`.

3. Implement binary cross-entropy loss from scratch and compute it for the
   full dataset using your model's predicted probabilities. Compare to a
   "dumb" model that always predicts `P=0.5` — which has lower loss?

4. Implement gradient descent for logistic regression completely from
   scratch (using the clean gradient formula `(ŷ-y)*x` from the concepts
   doc). Train on the same data and compare your learned coefficient to
   `sklearn`'s.

5. Plot the sigmoid curve `P(passed=1)` vs `hours_studied` from 0 to 10
   using your fitted coefficients, overlaid with a scatter of the actual
   data points (colored by `passed`). Does the curve visually match where
   passes/fails cluster?

6. Create a 2D dataset where two classes require a *curved* boundary (e.g.
   points in a small circle = class 1, points in a surrounding ring = class
   0). Fit plain logistic regression and report its accuracy — confirm it
   performs poorly, then add polynomial features
   (`PolynomialFeatures(degree=2)`) and refit — does accuracy improve?
