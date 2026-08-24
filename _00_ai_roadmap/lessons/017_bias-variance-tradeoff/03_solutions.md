# 03 — Solutions: Bias-Variance Tradeoff

```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
x = rng.uniform(-3, 3, 60)
y = 0.5 * x**3 - 2 * x + rng.normal(0, 3, 60)

x_train, x_test = x[:40], x[40:]
y_train, y_test = y[:40], y[40:]
```

## 1. Degree 1 vs degree 15

```python
def fit_and_eval(degree):
    coeffs = np.polyfit(x_train, y_train, degree)
    train_mse = np.mean((np.polyval(coeffs, x_train) - y_train) ** 2)
    test_mse = np.mean((np.polyval(coeffs, x_test) - y_test) ** 2)
    return coeffs, train_mse, test_mse

_, train1, test1 = fit_and_eval(1)
_, train15, test15 = fit_and_eval(15)
print("degree 1:", train1, test1)
print("degree 15:", train15, test15)
```

Degree 15 typically achieves much lower (sometimes near-zero) training error
but often *much higher* test error than degree 1 — the classic overfitting
signature: fitting noise in the 40 training points instead of the true cubic
trend, at the cost of generalization.

## 2. Visualizing under/overfit

```python
xs = np.linspace(-3, 3, 200)
c1, _, _ = fit_and_eval(1)
c15, _, _ = fit_and_eval(15)

plt.scatter(x_train, y_train, label="train data")
plt.plot(xs, np.polyval(c1, xs), label="degree 1 (underfit)")
plt.plot(xs, np.polyval(c15, xs), label="degree 15 (overfit)")
plt.ylim(y.min() - 5, y.max() + 5)   # degree-15 can swing wildly outside this
plt.legend()
plt.show()
```

The degree-1 line clearly misses the curvature; the degree-15 curve will
wiggle tightly through training points but swing wildly between them —
visually obvious overfitting.

## 3. MSE vs degree

```python
degrees = range(1, 16)
train_errs, test_errs = [], []
for d in degrees:
    _, tr, te = fit_and_eval(d)
    train_errs.append(tr)
    test_errs.append(te)

plt.plot(degrees, train_errs, label="train MSE")
plt.plot(degrees, test_errs, label="test MSE")
plt.yscale("log")
plt.legend()
plt.show()

best_degree = degrees[np.argmin(test_errs)]
print("best degree:", best_degree)
```

Training MSE should decrease monotonically (or nearly so) with degree; test
MSE should decrease then increase, forming a U-shape — the minimum of that
U is the empirical bias-variance sweet spot (often near degree 3, matching
the data's true cubic generating function, though noise can shift it).

## 4. Instability of a single split

```python
all_test_errs = []
for seed in range(5):
    rng2 = np.random.default_rng(seed)
    idx = rng2.permutation(60)
    xs_, ys_ = x[idx], y[idx]
    xtr, xte = xs_[:40], xs_[40:]
    ytr, yte = ys_[:40], ys_[40:]

    errs = []
    for d in degrees:
        coeffs = np.polyfit(xtr, ytr, d)
        errs.append(np.mean((np.polyval(coeffs, xte) - yte) ** 2))
    all_test_errs.append(errs)

avg_errs = np.mean(all_test_errs, axis=0)
print("best degree per split:", [degrees[np.argmin(e)] for e in all_test_errs])
print("best degree averaged:", degrees[np.argmin(avg_errs)])
```

The "best degree" often varies somewhat across individual splits (small
data, noisy estimate) but the averaged result is more stable — exactly why
relying on one train/test split for model selection is risky, and why
cross-validation (Q5) is preferred in practice.

## 5. 5-fold cross-validation for degree 3

```python
def kfold_indices(n, k, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    return np.array_split(idx, k)

folds = kfold_indices(60, 5)
val_mses = []
for i in range(5):
    val_idx = folds[i]
    train_idx = np.concatenate([folds[j] for j in range(5) if j != i])
    coeffs = np.polyfit(x[train_idx], y[train_idx], 3)
    val_mse = np.mean((np.polyval(coeffs, x[val_idx]) - y[val_idx]) ** 2)
    val_mses.append(val_mse)

print("mean:", np.mean(val_mses), "std:", np.std(val_mses))
```

## 6. Why 99% training accuracy isn't automatically good news

Q1–Q3 showed directly that training error can be driven arbitrarily low
(degree 15's training MSE) while test error gets *worse* — the model isn't
learning the true pattern, it's memorizing training-specific noise. Training
accuracy only tells you how well the model fits data it has already seen;
the entire point of a model is to perform well on data it *hasn't* seen,
which is exactly what a held-out test set (or cross-validation) measures
instead.
