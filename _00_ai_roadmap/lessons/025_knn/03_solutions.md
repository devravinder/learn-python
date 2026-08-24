# 03 — Solutions: K-Nearest Neighbors

```python
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

X, y = make_moons(n_samples=300, noise=0.3, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

## 1. k=1 vs k=15 vs k=100

```python
for k in [1, 15, 100]:
    model = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
    print(k, "train:", model.score(X_train, y_train), "test:", model.score(X_test, y_test))
```

`k=1` typically shows perfect or near-perfect training accuracy but lower
test accuracy — classic overfitting (every training point is its own
nearest neighbor). `k=100` (larger than a meaningful local neighborhood for
300 points) tends to underfit, smoothing away the moons' actual curved
structure. A middle `k` should generalize best.

## 2. k sweep

```python
import matplotlib.pyplot as plt

ks = range(1, 51)
train_accs, test_accs = [], []
for k in ks:
    m = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
    train_accs.append(m.score(X_train, y_train))
    test_accs.append(m.score(X_test, y_test))

plt.plot(ks, train_accs, label="train")
plt.plot(ks, test_accs, label="test")
plt.legend()
plt.show()
print("best k:", list(ks)[np.argmax(test_accs)])
```

## 3. KNN from scratch

```python
def knn_predict(X_train, y_train, query, k=5):
    dists = np.linalg.norm(X_train - query, axis=1)
    nearest_idx = np.argsort(dists)[:k]
    nearest_labels = y_train[nearest_idx]
    values, counts = np.unique(nearest_labels, return_counts=True)
    return values[np.argmax(counts)]

sklearn_model = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
for i in range(5):
    scratch_pred = knn_predict(X_train, y_train, X_test[i], k=5)
    sklearn_pred = sklearn_model.predict([X_test[i]])[0]
    print(scratch_pred, sklearn_pred, scratch_pred == sklearn_pred)
```

## 4. Effect of unscaled features

```python
rng = np.random.default_rng(0)
n = 400
feature_a = rng.uniform(0, 1, n)
feature_b = rng.uniform(0, 10000, n)   # irrelevant, huge scale
y2 = (feature_a > 0.5).astype(int)
X2 = np.column_stack([feature_a, feature_b])

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=0)

unscaled = KNeighborsClassifier(n_neighbors=5).fit(X2_train, y2_train)
print("unscaled accuracy:", unscaled.score(X2_test, y2_test))

scaler = StandardScaler().fit(X2_train)
scaled_model = KNeighborsClassifier(n_neighbors=5).fit(scaler.transform(X2_train), y2_train)
print("scaled accuracy:", scaled_model.score(scaler.transform(X2_test), y2_test))
```

Unscaled, `feature_b`'s huge range dominates the Euclidean distance
calculation even though it's pure noise, dragging accuracy down toward
chance (~50%); after scaling, both features contribute comparably to
distance, and accuracy should recover close to 100% since `feature_a` alone
perfectly determines the label.

## 5. Distance metric comparison

```python
best_k = 15  # example, use your actual Q2 result
for metric in ["euclidean", "manhattan"]:
    m = KNeighborsClassifier(n_neighbors=best_k, metric=metric).fit(X_train, y_train)
    print(metric, m.score(X_test, y_test))
```

On this 2D dataset the two metrics usually perform similarly — the choice of
metric tends to matter more in higher-dimensional or more structured
feature spaces (e.g. text/embedding data, Lesson 010).

## 6. Prediction time vs training set size

```python
import time

for n in [500, 5000, 20000]:
    Xn, yn = make_moons(n_samples=n, noise=0.3, random_state=0)
    model = KNeighborsClassifier(n_neighbors=5).fit(Xn, yn)
    start = time.time()
    model.predict(Xn[:100])
    print(n, time.time() - start)
```

Prediction time should increase noticeably as training set size grows,
consistent with KNN needing to compare against every training point (or at
least search a tree built over all of them) at prediction time — unlike a
trained linear/logistic regression model, whose prediction cost doesn't
depend on training set size at all.
