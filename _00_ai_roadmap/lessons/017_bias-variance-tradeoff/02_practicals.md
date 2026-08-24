# 02 — Practicals: Bias-Variance Tradeoff

Use this dataset (true relationship is a cubic curve, not a line):

```python
import numpy as np
rng = np.random.default_rng(0)
x = rng.uniform(-3, 3, 60)
y = 0.5 * x**3 - 2 * x + rng.normal(0, 3, 60)

x_train, x_test = x[:40], x[40:]
y_train, y_test = y[:40], y[40:]
```

1. Fit a degree-1 polynomial (`np.polyfit(x_train, y_train, 1)`) and a
   degree-15 polynomial to the training data. For each, compute training MSE
   and test MSE (`np.polyval` to predict). Which has lower training error?
   Which has lower *test* error?

2. Plot both fitted curves against the training data scatter (use
   `np.linspace(-3, 3, 200)` for a smooth curve). Visually identify which
   model underfits and which overfits.

3. Fit polynomials of degree 1 through 15 and plot training MSE and test MSE
   against degree on the same chart. Identify the degree that minimizes test
   MSE — that's your empirical "sweet spot" for this data.

4. Repeat Q3 but generate 5 different random train/test splits (different
   `rng` seeds) and average the test MSE per degree. Does the "best degree"
   change between individual splits and the averaged result? What does that
   tell you about trusting a single train/test split?

5. Implement simple K-fold cross-validation (K=5) from scratch (no
   `sklearn.model_selection`) for the degree-3 polynomial: split the full
   dataset (`x`, `y`, 60 points) into 5 folds, train on 4 and validate on 1,
   rotate, and report the mean and standard deviation of validation MSE
   across the 5 folds.

6. Explain, using your Q1–Q3 results as a concrete example, why "my model
   gets 99% accuracy on training data" is not, by itself, good news.
