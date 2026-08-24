# 03 — Solutions: Multiple Linear Regression & Assumptions

## 1. VIF

```python
import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

rng = np.random.default_rng(0)
x1 = rng.normal(0, 1, 200)
x2 = x1 * 0.98 + rng.normal(0, 0.05, 200)
x3 = rng.normal(0, 1, 200)
y = 3*x1 + 2*x3 + rng.normal(0, 1, 200)
X = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})

vif = pd.DataFrame({
    "feature": X.columns,
    "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
})
print(vif)
```

`x1` and `x2` should both show very high VIF (often in the hundreds), since
each is almost perfectly predictable from the other; `x3` (independent)
should show a VIF close to 1.

## 2. Coefficient instability under multicollinearity

```python
from sklearn.linear_model import LinearRegression

full = LinearRegression().fit(X, y)
print("with x2:", full.coef_)

dropped = LinearRegression().fit(X[["x1", "x3"]], y)
print("without x2:", dropped.coef_)
```

The coefficient on `x1` can shift substantially (sometimes even change sign
in more extreme collinearity) between the two fits, because with `x1` and
`x2` both present, the model can't tell which one "deserves credit" for
their shared predictive signal — a concrete illustration of why individual
coefficients aren't trustworthy under high multicollinearity, even though
overall predictions may barely change.

## 3. Breusch-Pagan test

```python
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan

X_sm = sm.add_constant(X[["x1", "x3"]])
model = sm.OLS(y, X_sm).fit()
bp_test = het_breuschpagan(model.resid, X_sm)
print("p-value:", bp_test[1])
```

With this synthetic data (constant-variance noise by construction), the
p-value should typically be well above 0.05 — no evidence of
heteroscedasticity, as expected since the generator didn't introduce any.

## 4. One-hot encoding interpretation

```python
df = pd.DataFrame({
    "region": rng.choice(["North", "South", "East"], 200),
    "spend": rng.uniform(100, 1000, 200),
})
df["sales"] = df["spend"] * 0.5 + rng.normal(0, 50, 200)
df_encoded = pd.get_dummies(df, columns=["region"], drop_first=True)

X = df_encoded.drop(columns="sales")
model = LinearRegression().fit(X, df["sales"])
print(X.columns, model.coef_, model.intercept_)
```

With `drop_first=True`, one category (e.g. "East", alphabetically first)
becomes the **reference level**, absorbed into the intercept. Each remaining
dummy's coefficient represents "the additional effect of being in this
region *compared to the reference region*," not an absolute effect — a
common point of confusion when interpreting one-hot encoded regression
output.

## 5. Interaction term

```python
n = 200
discount = rng.uniform(0, 0.3, n)
is_holiday = rng.integers(0, 2, n)
sales = 100 + 50 * discount + 30 * is_holiday + 200 * discount * is_holiday + rng.normal(0, 5, n)

df = pd.DataFrame({"discount": discount, "is_holiday": is_holiday, "sales": sales})

X_no_interact = df[["discount", "is_holiday"]]
X_interact = X_no_interact.copy()
X_interact["discount_x_holiday"] = discount * is_holiday

from sklearn.metrics import r2_score
m1 = LinearRegression().fit(X_no_interact, sales)
m2 = LinearRegression().fit(X_interact, sales)

print("R2 without interaction:", r2_score(sales, m1.predict(X_no_interact)))
print("R2 with interaction:", r2_score(sales, m2.predict(X_interact)))
```

Since the data was generated *with* a genuine interaction effect (discount's
impact is much larger during holidays), the model without the interaction
term should show a noticeably lower R² — it can only fit an "average"
discount effect that's wrong for both holiday and non-holiday cases, while
the interaction model captures the true effect-depends-on-context
relationship directly.
