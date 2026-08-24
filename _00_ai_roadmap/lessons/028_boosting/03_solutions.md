# 03 — Solutions: Boosting

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier

X, y = make_classification(n_samples=1000, n_features=20, n_informative=8, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

## 1. Comparing ensemble types

```python
ada = AdaBoostClassifier(n_estimators=100, random_state=0).fit(X_train, y_train)
gb = GradientBoostingClassifier(n_estimators=100, random_state=0).fit(X_train, y_train)
rf = RandomForestClassifier(n_estimators=100, random_state=0).fit(X_train, y_train)

print("AdaBoost:", ada.score(X_test, y_test))
print("GradientBoosting:", gb.score(X_test, y_test))
print("RandomForest:", rf.score(X_test, y_test))
```

Gradient Boosting typically edges out both AdaBoost and Random Forest on
data like this when using reasonable defaults, consistent with the general
"boosting usually wins when tuned" pattern — though the margin varies by
dataset and none of these should be assumed to always win without checking.

## 2. Learning rate sweep

```python
lrs = [0.01, 0.05, 0.1, 0.5, 1.0]
accs = []
for lr in lrs:
    m = GradientBoostingClassifier(n_estimators=100, learning_rate=lr, random_state=0)
    m.fit(X_train, y_train)
    accs.append(m.score(X_test, y_test))

plt.plot(lrs, accs, marker="o")
plt.xscale("log")
plt.show()
```

Very high learning rates (e.g. 1.0) often hurt test accuracy compared to a
moderate rate (e.g. 0.1) — each tree overcorrects, similar in spirit to
Lesson 015's "too-large learning rate overshoots" for gradient descent in
parameter space, here happening in function space instead.

## 3. n_estimators sweep at fixed low learning rate

```python
n_list = [10, 50, 100, 300, 500]
train_accs, test_accs = [], []
for n in n_list:
    m = GradientBoostingClassifier(n_estimators=n, learning_rate=0.05, random_state=0)
    m.fit(X_train, y_train)
    train_accs.append(m.score(X_train, y_train))
    test_accs.append(m.score(X_test, y_test))

plt.plot(n_list, train_accs, label="train")
plt.plot(n_list, test_accs, label="test")
plt.xscale("log")
plt.legend()
plt.show()
```

With a low learning rate, overfitting tends to show up only at very high
`n_estimators` (if at all within this range) — train accuracy keeps
climbing toward 1.0 while test accuracy plateaus or very slowly declines,
the same underlying bias-variance pattern as Lesson 017, paced more slowly
by the small learning rate.

## 4. Early stopping

```python
# with xgboost:
import xgboost as xgb
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)
model = xgb.XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=3, early_stopping_rounds=10, eval_metric="logloss")
model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
print("trees actually used:", model.best_iteration)

# manual equivalent without xgboost:
gb = GradientBoostingClassifier(n_estimators=500, learning_rate=0.05, random_state=0)
gb.fit(X_tr, y_tr)
val_accs = [np.mean(pred == y_val) for pred in gb.staged_predict(X_val)]
best_round = int(np.argmax(val_accs))
print("best round (manual):", best_round, "of", len(val_accs))
```

The model typically stops well short of the full requested `n_estimators`
once validation performance plateaus — directly demonstrating why early
stopping saves both compute and overfitting risk compared to a fixed,
possibly-too-large tree count.

## 5. Training time: parallel vs sequential

```python
import time

t0 = time.time()
RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1).fit(X_train, y_train)
print("RF time:", time.time() - t0)

t0 = time.time()
GradientBoostingClassifier(n_estimators=200, random_state=0).fit(X_train, y_train)
print("GB time:", time.time() - t0)
```

Random Forest is typically faster (especially with `n_jobs=-1` enabling
parallel tree building across CPU cores), because its trees are independent
and can be built simultaneously; Gradient Boosting's trees must be built
one after another (each depends on the previous ensemble's residuals),
making it inherently sequential and usually slower to train for the same
tree count.

## 6. Feature importance agreement

```python
gb_importance = gb.feature_importances_
rf_importance = rf.feature_importances_

top3_gb = np.argsort(gb_importance)[-3:]
top3_rf = np.argsort(rf_importance)[-3:]
print("GB top 3:", sorted(top3_gb))
print("RF top 3:", sorted(top3_rf))
```

The two ensemble types usually agree substantially on the top few most
important features (both are, after all, built from decision trees on the
same data), though the exact ranking/magnitudes can differ due to their
different training mechanics (sequential residual-fitting vs independent
bootstrap averaging) — general agreement with some difference in emphasis
is the expected, healthy outcome.
