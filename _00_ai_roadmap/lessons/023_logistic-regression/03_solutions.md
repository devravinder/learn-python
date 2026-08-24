# 03 — Solutions: Logistic Regression

```python
import numpy as np
from sklearn.linear_model import LogisticRegression

rng = np.random.default_rng(0)
n = 300
hours_studied = rng.uniform(0, 10, n)
z = 1.5 * hours_studied - 7 + rng.normal(0, 1, n)
passed = (z > 0).astype(int)

X = hours_studied.reshape(-1, 1)
```

## 1. Fit

```python
model = LogisticRegression().fit(X, passed)
print(model.coef_, model.intercept_)   # should be roughly close to [1.5], [-7]
```

## 2. Manual sigmoid check

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

w, b = model.coef_[0, 0], model.intercept_[0]
manual_prob = sigmoid(w * 5 + b)
sklearn_prob = model.predict_proba([[5]])[0, 1]
print(manual_prob, sklearn_prob)   # match
```

## 3. Binary cross-entropy vs dumb baseline

```python
def bce(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

probs = model.predict_proba(X)[:, 1]
print("model loss:", bce(passed, probs))
print("dumb (p=0.5) loss:", bce(passed, np.full(n, 0.5)))
```

The dumb baseline's loss is exactly `log(2) ≈ 0.693` regardless of the
data; the trained model's loss should be substantially lower, since it
actually uses `hours_studied` to make confident, mostly-correct predictions.

## 4. Logistic regression from scratch

```python
def train_logistic(X, y, lr=0.1, epochs=2000):
    w = np.zeros(X.shape[1])
    b = 0.0
    n = len(y)
    for _ in range(epochs):
        z = X @ w + b
        y_hat = sigmoid(z)
        error = y_hat - y
        dw = (1 / n) * (X.T @ error)
        db = (1 / n) * error.sum()
        w -= lr * dw
        b -= lr * db
    return w, b

w_scratch, b_scratch = train_logistic(X, passed)
print(w_scratch, b_scratch, "vs sklearn:", model.coef_, model.intercept_)
```

Both should converge to similar coefficients, up to sklearn's default
regularization (sklearn's `LogisticRegression` applies L2 regularization by
default with `C=1.0`; setting `penalty=None` in sklearn would match the
unregularized from-scratch version more closely).

## 5. Sigmoid curve plot

```python
import matplotlib.pyplot as plt

xs = np.linspace(0, 10, 200)
probs_curve = sigmoid(w * xs + b)

plt.plot(xs, probs_curve, color="black")
plt.scatter(hours_studied, passed, c=passed, cmap="coolwarm", alpha=0.5)
plt.xlabel("hours studied")
plt.ylabel("P(passed)")
plt.show()
```

The S-curve should cross 0.5 right around `hours_studied ≈ 7/1.5 ≈ 4.7`,
matching where the data visibly transitions from mostly-0 to mostly-1.

## 6. Nonlinear boundary requires nonlinear features

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

rng = np.random.default_rng(1)
theta = rng.uniform(0, 2*np.pi, 200)
r_inner = rng.normal(1, 0.2, 100)
r_outer = rng.normal(3, 0.2, 100)

x_inner = np.column_stack([r_inner*np.cos(theta[:100]), r_inner*np.sin(theta[:100])])
x_outer = np.column_stack([r_outer*np.cos(theta[100:]), r_outer*np.sin(theta[100:])])
X2 = np.vstack([x_inner, x_outer])
y2 = np.array([1]*100 + [0]*100)

plain = LogisticRegression().fit(X2, y2)
print("plain accuracy:", plain.score(X2, y2))   # likely poor, near 50%

poly_model = make_pipeline(PolynomialFeatures(2), LogisticRegression())
poly_model.fit(X2, y2)
print("with poly features accuracy:", poly_model.score(X2, y2))   # much better
```

Plain logistic regression can only draw a straight line, which can't
separate a circle from a surrounding ring (any line cuts through both
groups); adding degree-2 polynomial features gives the model access to `x²`,
`y²` terms, letting it effectively learn a circular decision boundary and
dramatically improving accuracy.
