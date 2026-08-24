# 01 — Concepts: Polynomial Regression & Regularization

## Polynomial regression is still linear regression

`ŷ = w1*x + w2*x^2 + w3*x^3 + b` is linear in the *weights* `w1, w2, w3`,
even though it's nonlinear in `x` — so it's fit with the exact same
machinery (Lesson 020), just after adding polynomial feature columns:

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

model = make_pipeline(PolynomialFeatures(degree=3), LinearRegression())
model.fit(X, y)
```

As you saw in Lesson 017, high-degree polynomials overfit easily — this is
exactly where regularization comes in.

## Regularization: penalize large weights

Add a penalty term to the loss that discourages large coefficients:

```
Ridge:  Loss = MSE + α * Σ w_i^2          (L2 penalty)
Lasso:  Loss = MSE + α * Σ |w_i|           (L1 penalty)
```

`α` controls the strength: `α=0` recovers plain linear regression;
large `α` shrinks weights aggressively (more bias, less variance —
Lesson 017's tradeoff, dialed explicitly).

```python
from sklearn.linear_model import Ridge, Lasso

ridge = Ridge(alpha=1.0).fit(X_train, y_train)
lasso = Lasso(alpha=1.0).fit(X_train, y_train)
```

## Why L1 gives sparsity and L2 doesn't

Geometrically: L1's penalty region (a diamond in 2D, a cross-polytope in
higher dimensions) has sharp corners *on the axes*, so the optimal solution
often lands exactly on a corner — meaning some weights become **exactly
zero**. L2's penalty region (a circle/sphere) is smooth everywhere, so
weights shrink toward zero but rarely hit it exactly.

**Practical consequence**: Lasso performs automatic **feature selection**
(irrelevant features get a coefficient of exactly 0); Ridge shrinks all
coefficients but keeps every feature in the model, which tends to work
better when features are correlated (Lesson 021's multicollinearity —
Ridge spreads weight across correlated features rather than arbitrarily
picking one, as Lasso tends to).

## Elastic Net

A weighted combination of both penalties (`α * [ρ * L1 + (1-ρ) * L2]`) —
gets some sparsity from L1 while keeping L2's better handling of correlated
features. `sklearn.linear_model.ElasticNet`.

## Choosing α via cross-validation

`α` is a hyperparameter — tuned via cross-validation (Lesson 017), not
learned from the training data directly (using the same data to both fit and
tune would just re-introduce overfitting through the back door).
`RidgeCV`/`LassoCV` handle this search automatically:

```python
from sklearn.linear_model import RidgeCV
model = RidgeCV(alphas=[0.01, 0.1, 1, 10, 100], cv=5).fit(X_train, y_train)
print(model.alpha_)
```

## Feature scaling matters even more here

Regularization penalizes coefficient *magnitude* directly — if features are
on wildly different scales, their coefficients will naturally be on
different scales too (a feature measured in millions needs a tiny
coefficient; one measured in single digits needs a large one), so the
penalty would unfairly punish features with naturally large-scale units.
**Always standardize features before Ridge/Lasso.**
