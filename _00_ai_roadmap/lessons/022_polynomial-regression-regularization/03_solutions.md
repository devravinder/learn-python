# 03 — Solutions: Polynomial Regression & Regularization

```python
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

rng = np.random.default_rng(0)
x = rng.uniform(-3, 3, 60).reshape(-1, 1)
y = 0.5 * x.ravel()**3 - 2 * x.ravel() + rng.normal(0, 3, 60)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
```

## 1. Unregularized degree-15 polynomial

```python
model = make_pipeline(PolynomialFeatures(15), LinearRegression())
model.fit(x_train, y_train)
mse_plain = mean_squared_error(y_test, model.predict(x_test))
print(mse_plain)   # likely large - overfit
```

## 2. Ridge-regularized degree-15

```python
ridge_model = make_pipeline(PolynomialFeatures(15), Ridge(alpha=1.0))
ridge_model.fit(x_train, y_train)
mse_ridge = mean_squared_error(y_test, ridge_model.predict(x_test))
print(mse_ridge)   # typically much lower than mse_plain
```

Ridge should substantially reduce test MSE compared to the unregularized
degree-15 fit, since it constrains the wild coefficient swings that would
otherwise let the polynomial chase noise between training points.

## 3. Alpha sweep

```python
import matplotlib.pyplot as plt

alphas = [0.001, 0.01, 0.1, 1, 10, 100]
mses = []
for a in alphas:
    m = make_pipeline(PolynomialFeatures(15), Ridge(alpha=a))
    m.fit(x_train, y_train)
    mses.append(mean_squared_error(y_test, m.predict(x_test)))

plt.plot(alphas, mses, marker="o")
plt.xscale("log")
plt.show()
print(alphas[np.argmin(mses)])
```

Expect a U-shape: too-small alpha barely regularizes (still overfits close
to the plain model); too-large alpha over-shrinks everything toward 0
(underfits); a middle value minimizes test MSE.

## 4. Lasso sparsity

```python
rng = np.random.default_rng(1)
X = rng.normal(size=(200, 20))
true_weights = np.zeros(20)
true_weights[[2, 7, 15]] = [5, -3, 2]
y2 = X @ true_weights + rng.normal(0, 1, 200)

lasso = Lasso(alpha=0.5).fit(X, y2)
print(lasso.coef_.round(2))
print("nonzero indices:", np.nonzero(np.abs(lasso.coef_) > 1e-3)[0])
```

Lasso should zero out most of the 17 irrelevant coefficients exactly (or
very close to it) and keep nonzero coefficients close to indices 2, 7, 15 —
directly recovering which features were actually used to generate `y`.

## 5. Ridge doesn't zero out

```python
ridge = Ridge(alpha=0.5).fit(X, y2)
print(ridge.coef_.round(3))
```

Ridge's coefficients for the 17 irrelevant features should be small but
**not exactly zero** — visibly different from Lasso's behavior, confirming
the "L1 gives sparsity, L2 shrinks smoothly" distinction from the concepts
doc.

## 6. Effect of standardization on wildly-scaled features

```python
import pandas as pd
housing = pd.read_csv("../../../projects/002_gradient-descent-housing/02_solutions/data/housing.csv")
Xh = housing[["sqft", "bedrooms", "age", "distance_km"]].to_numpy()
yh = housing["price"].to_numpy()

ridge_raw = Ridge(alpha=1.0).fit(Xh, yh)
print("raw coefficients:", ridge_raw.coef_)

Xh_std = (Xh - Xh.mean(axis=0)) / Xh.std(axis=0)
ridge_std = Ridge(alpha=1.0).fit(Xh_std, yh)
print("standardized coefficients:", ridge_std.coef_)
```

On raw features, `sqft` (scale ~500-4000) gets an artificially tiny
coefficient purely because of its scale, while `bedrooms` (scale 1-6) gets a
much larger one — the same `alpha` penalizes them very unevenly relative to
their actual importance. After standardizing, coefficient magnitudes
directly reflect each feature's real importance, which is why standardizing
before Ridge/Lasso is standard practice, not optional polish.
