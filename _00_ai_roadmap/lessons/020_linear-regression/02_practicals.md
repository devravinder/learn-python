# 02 — Practicals: Linear Regression

Reuse Project 002's `housing.csv` (regenerate via its `generate_data.py` if
needed), or this quick equivalent:

```python
import numpy as np
import pandas as pd
rng = np.random.default_rng(3)
n = 300
df = pd.DataFrame({
    "sqft": rng.uniform(500, 4000, n),
    "bedrooms": rng.integers(1, 6, n),
    "age": rng.uniform(0, 50, n),
})
df["price"] = 50000 + 150*df.sqft + 10000*df.bedrooms - 800*df.age + rng.normal(0, 15000, n)
```

1. Split into train/test (80/20), fit `sklearn.linear_model.LinearRegression`,
   and report R², MAE, and RMSE on the test set.

2. Print `model.coef_` and `model.intercept_`. Do the learned coefficients
   roughly match the true generating coefficients (150, 10000, -800)?

3. Plot residuals (`y_test - predictions`) against predicted values. Do they
   look randomly scattered, or is there a visible pattern?

4. Compute R² and adjusted R² by hand from the formulas
   (`adj_r2 = 1 - (1-r2)*(n-1)/(n-p-1)`, `p` = number of features). Add 5
   random noise columns to the features, refit, and compare R² vs adjusted R²
   — does adjusted R² correctly fail to reward the useless noise features
   while plain R² still increases (even if slightly)?

5. Fit a model using only `sqft` (drop bedrooms/age) and compare its test R²
   to the full model's. Quantify how much predictive power you lose by
   dropping two real, relevant features.
