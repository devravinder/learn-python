# 03 — Solutions: Linear Regression

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

rng = np.random.default_rng(3)
n = 300
df = pd.DataFrame({
    "sqft": rng.uniform(500, 4000, n),
    "bedrooms": rng.integers(1, 6, n),
    "age": rng.uniform(0, 50, n),
})
df["price"] = 50000 + 150*df.sqft + 10000*df.bedrooms - 800*df.age + rng.normal(0, 15000, n)

X = df[["sqft", "bedrooms", "age"]]
y = df["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

## 1–2. Fit and inspect coefficients

```python
model = LinearRegression().fit(X_train, y_train)
preds = model.predict(X_test)

print("R2:", r2_score(y_test, preds))
print("MAE:", mean_absolute_error(y_test, preds))
print("RMSE:", mean_squared_error(y_test, preds) ** 0.5)
print("coefficients:", model.coef_, "intercept:", model.intercept_)
```

Coefficients should land close to `[150, 10000, -800]` since that's exactly
the data-generating process, with some deviation from the added noise.

## 3. Residual plot

```python
import matplotlib.pyplot as plt

residuals = y_test - preds
plt.scatter(preds, residuals)
plt.axhline(0, color="red")
plt.xlabel("predicted")
plt.ylabel("residual")
plt.show()
```

Residuals should scatter randomly around 0 with no visible curve or funnel
shape, since the true relationship really is linear here — a real dataset
with a nonlinear relationship would show a visible pattern instead.

## 4. R² vs adjusted R²

```python
def adjusted_r2(r2, n, p):
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

r2_full = r2_score(y_test, preds)
adj_full = adjusted_r2(r2_full, len(y_test), X_test.shape[1])

noise = rng.normal(0, 1, size=(n, 5))
X_noisy = np.column_stack([X.to_numpy(), noise])
Xn_train, Xn_test, yn_train, yn_test = train_test_split(X_noisy, y, test_size=0.2, random_state=0)

model_noisy = LinearRegression().fit(Xn_train, yn_train)
preds_noisy = model_noisy.predict(Xn_test)
r2_noisy = r2_score(yn_test, preds_noisy)
adj_noisy = adjusted_r2(r2_noisy, len(yn_test), Xn_test.shape[1])

print("plain R2:  full", r2_full, "noisy", r2_noisy)
print("adj R2:    full", adj_full, "noisy", adj_noisy)
```

Plain R² typically ticks up very slightly even with pure noise features
(it can never decrease from adding features on the *training* set, and only
rarely decreases meaningfully on a held-out test set by chance); adjusted R²
should stay flat or decrease, correctly reflecting that the noise features
add no real predictive value once penalized for the added complexity.

## 5. Dropping real features

```python
X_sqft_only = df[["sqft"]]
Xs_train, Xs_test, ys_train, ys_test = train_test_split(X_sqft_only, y, test_size=0.2, random_state=0)
model_simple = LinearRegression().fit(Xs_train, ys_train)
r2_simple = r2_score(ys_test, model_simple.predict(Xs_test))

print("full model R2:", r2_full, "sqft-only R2:", r2_simple)
```

Dropping `bedrooms` and `age` (both genuinely predictive by construction)
should noticeably reduce R² compared to the full model — quantifying the
real cost of omitting relevant features, as opposed to Q4's noise features
which cost little to nothing to omit.
