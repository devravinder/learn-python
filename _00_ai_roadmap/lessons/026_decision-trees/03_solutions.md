# 03 — Solutions: Decision Trees

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

X, y = make_classification(n_samples=500, n_features=6, n_informative=4, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

## 1. Unconstrained tree

```python
model = DecisionTreeClassifier(random_state=0).fit(X_train, y_train)
print("train:", model.score(X_train, y_train))   # often 1.0
print("test:", model.score(X_test, y_test))       # noticeably lower
```

A perfect (or near-perfect) training accuracy alongside a clearly lower test
accuracy is the textbook overfitting signature from Lesson 017 — an
unconstrained tree memorizes training data down to individual points.

## 2. Depth sweep

```python
import matplotlib.pyplot as plt

depths = range(1, 16)
train_accs, test_accs = [], []
for d in depths:
    m = DecisionTreeClassifier(max_depth=d, random_state=0).fit(X_train, y_train)
    train_accs.append(m.score(X_train, y_train))
    test_accs.append(m.score(X_test, y_test))

plt.plot(depths, train_accs, label="train")
plt.plot(depths, test_accs, label="test")
plt.legend()
plt.show()
print("best depth:", list(depths)[np.argmax(test_accs)])
```

## 3. Gini from scratch

```python
def gini(labels):
    labels = np.asarray(labels)
    _, counts = np.unique(labels, return_counts=True)
    p = counts / counts.sum()
    return 1 - np.sum(p ** 2)

print(gini([0, 0, 1, 1]))   # 0.5 (max for 2 classes)
print(gini([1, 1, 1, 1]))   # 0.0 (pure)
```

## 4. Information gain from scratch

```python
def entropy(labels):
    labels = np.asarray(labels)
    _, counts = np.unique(labels, return_counts=True)
    p = counts / counts.sum()
    p = p[p > 0]
    return -np.sum(p * np.log2(p))

def information_gain(parent, left, right):
    n = len(parent)
    weighted_child_entropy = (len(left)/n)*entropy(left) + (len(right)/n)*entropy(right)
    return entropy(parent) - weighted_child_entropy

# toy: feature_perfect splits classes exactly, feature_random doesn't
labels = np.array([0,0,0,1,1,1])
perfect_left, perfect_right = labels[:3], labels[3:]        # [0,0,0] | [1,1,1]
random_left, random_right = labels[[0,1,3]], labels[[2,4,5]]  # [0,0,1] | [0,1,1]

print("perfect split gain:", information_gain(labels, perfect_left, perfect_right))  # 1.0 (max)
print("random split gain:", information_gain(labels, random_left, random_right))     # much lower
```

## 5. Shallow tree on Titanic-style data

```python
import pandas as pd
from sklearn.tree import plot_tree

df = pd.read_csv("../../../assignments/002_classification-titanic-style/02_solutions/data/titanic_synthetic.csv")
df["age"] = df.groupby("pclass")["age"].transform(lambda s: s.fillna(s.median()))
df = pd.get_dummies(df, columns=["sex"], drop_first=True)

features = ["pclass", "sex_male", "age", "fare", "sibsp"]
X, y = df[features], df["survived"]
tree = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X, y)

plt.figure(figsize=(14, 8))
plot_tree(tree, feature_names=features, filled=True, class_names=["died", "survived"])
plt.show()
```

The root split typically picks `sex_male` first, matching logistic
regression's finding (Assignment 002) that sex is the strongest predictor of
survival in this synthetic dataset — a good cross-check that two very
different model types agree on what matters most.

## 6. Feature importance vs correlation

```python
importances = dict(zip(features, tree.feature_importances_))
correlations = {f: abs(np.corrcoef(X[f], y)[0, 1]) for f in features}

print("feature importances:", importances)
print("|correlation| with target:", correlations)
```

The rankings usually roughly agree (the same 1-2 features dominate both),
though trees can capture non-linear/interaction effects correlation can't
see, and correlation can flag a linearly-related feature the tree
under-uses if a correlated feature was split on first — the two aren't
identical, complementary views of "importance," not interchangeable.
