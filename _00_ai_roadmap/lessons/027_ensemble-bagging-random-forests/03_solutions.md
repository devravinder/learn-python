# 03 — Solutions: Bagging & Random Forests

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier

X, y = make_classification(
    n_samples=500, n_features=15, n_informative=5, n_redundant=5,
    random_state=0,
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

## 1. Single tree vs forest

```python
tree = DecisionTreeClassifier(random_state=0).fit(X_train, y_train)
forest = RandomForestClassifier(n_estimators=200, random_state=0).fit(X_train, y_train)

print("tree:", tree.score(X_test, y_test))
print("forest:", forest.score(X_test, y_test))
```

The forest should outperform the single tree by a meaningful margin on most
runs of data like this — the expected outcome, since it directly attacks the
tree's main weakness (variance).

## 2. OOB score vs held-out test accuracy

```python
forest_oob = RandomForestClassifier(n_estimators=200, oob_score=True, random_state=0)
forest_oob.fit(X_train, y_train)
print("OOB score:", forest_oob.oob_score_)
print("test score:", forest_oob.score(X_test, y_test))
```

These should land close to each other. OOB score is legitimate because each
tree in the forest is validated only on the ~37% of training samples it
never saw during its own bootstrap sampling — functionally equivalent to a
held-out validation set, but obtained "for free" without sacrificing any
training data.

## 3. n_estimators sweep

```python
import matplotlib.pyplot as plt

n_trees_list = [1, 5, 20, 50, 100, 300]
accs = []
for n in n_trees_list:
    m = RandomForestClassifier(n_estimators=n, random_state=0).fit(X_train, y_train)
    accs.append(m.score(X_test, y_test))

plt.plot(n_trees_list, accs, marker="o")
plt.xscale("log")
plt.show()
```

Accuracy typically rises sharply from 1 to ~20-50 trees, then plateaus —
diminishing returns kick in once you have "enough" trees for the averaging
effect to stabilize; adding hundreds more rarely hurts but stops helping
much.

## 4. Bagging vs Random Forest

```python
bagging = BaggingClassifier(DecisionTreeClassifier(), n_estimators=200, random_state=0)
bagging.fit(X_train, y_train)
rf = RandomForestClassifier(n_estimators=200, random_state=0).fit(X_train, y_train)

print("bagging:", bagging.score(X_test, y_test))
print("random forest:", rf.score(X_test, y_test))
```

Random Forest often edges out plain bagging, especially when there are
several correlated/redundant features (this dataset has 5 redundant ones by
construction) — exactly the situation where feature subsetting's
decorrelation benefit matters most, though the margin can be small.

## 5. Feature importance comparison

```python
tree_importance = tree.feature_importances_
forest_importance = forest.feature_importances_

print("tree:", tree_importance.round(3))
print("forest:", forest_importance.round(3))
```

The forest's importances are generally more trustworthy: a single tree's
importances depend heavily on the arbitrary choices made during its one
greedy growth process (Lesson 026), while the forest's are averaged across
many differently-grown trees, smoothing out that single-tree noise.

## 6. Robustness to label noise

```python
rng = np.random.default_rng(0)
y_train_noisy = y_train.copy()
flip_idx = rng.choice(len(y_train), size=int(0.1 * len(y_train)), replace=False)
y_train_noisy[flip_idx] = 1 - y_train_noisy[flip_idx]

tree_noisy = DecisionTreeClassifier(random_state=0).fit(X_train, y_train_noisy)
forest_noisy = RandomForestClassifier(n_estimators=200, random_state=0).fit(X_train, y_train_noisy)

print("tree with noise:", tree_noisy.score(X_test, y_test))
print("forest with noise:", forest_noisy.score(X_test, y_test))
```

The single tree typically degrades more than the forest: a tree will
directly incorporate every mislabeled point into its splits (it has no
mechanism to "outvote" bad data), while the forest's trees each see a
different bootstrap sample, so a given noisy label only pollutes a subset of
trees, and majority voting across the whole ensemble dilutes its effect —
concretely demonstrating the "averaging cancels uncorrelated errors"
argument, applied to noisy labels instead of just to variance from data
sampling.
