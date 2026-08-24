# 01 — Concepts: Linear Regression

## The model

```
ŷ = w1*x1 + w2*x2 + ... + wn*xn + b
```

A weighted sum of features plus a bias — the simplest possible supervised
model, and the direct ancestor of every neural network layer (Lesson 035).

## Fitting it: closed-form vs gradient descent

You already did both in Project 002. In practice: `sklearn`'s
`LinearRegression` uses the closed-form/least-squares solution (fast, exact,
fine for small-to-medium data); gradient descent (`SGDRegressor`, or any
neural net) is needed once data is too large for a matrix inverse, or once
the model isn't linear-in-closed-form (everything from Lesson 035 onward).

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(model.coef_, model.intercept_)
```

## Residuals — the difference between prediction and reality

`residual_i = y_i - ŷ_i`. Plotting residuals against predicted values (or
against each feature) is the standard diagnostic:

- **Residuals scattered randomly around 0** → good fit, no obvious pattern
  left unexplained.
- **Residuals show a curve/pattern** → the true relationship isn't linear;
  consider polynomial terms (Lesson 022) or a different model.
- **Residual spread grows with predicted value** (a "funnel" shape) →
  heteroscedasticity, violates a linear regression assumption (see below).

## Assumptions behind linear regression (full detail in Lesson 021)

1. **Linearity**: the true relationship is (approximately) linear in the
   parameters.
2. **Independence**: residuals aren't correlated with each other (a problem
   for time-series data, where today's residual often relates to
   yesterday's).
3. **Homoscedasticity**: residual variance is constant across all predicted
   values (no funnel shape).
4. **Normality of residuals**: mostly matters for the validity of confidence
   intervals/p-values on coefficients, less for pure prediction accuracy.

Violating these doesn't necessarily make predictions useless, but it does
undermine any statistical claims (confidence intervals, significance tests)
you might make about the coefficients.

## Interpreting coefficients

`w_i` is "the change in `y` for a one-unit change in `x_i`, holding all other
features constant." This holding-constant interpretation only makes sense if
features aren't strongly correlated with each other (see multicollinearity,
Lesson 021) — otherwise "holding others constant" describes situations that
barely occur in your actual data.

## R² and adjusted R²

R² (Lesson 018) always increases (or stays the same) as you add more
features, even useless ones — it rewards model complexity for free.
**Adjusted R²** penalizes each added feature, only increasing if the new
feature improves fit by more than chance would predict — a fairer comparison
across models with different numbers of features.
