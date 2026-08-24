# 03 — Solutions: Support Vector Machines

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_circles
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
```

## 1. Linear SVM and support vectors

```python
X_lin, y_lin = make_blobs(n_samples=200, centers=2, cluster_std=1.5, random_state=0)
model = SVC(kernel="linear").fit(X_lin, y_lin)

plt.scatter(X_lin[:, 0], X_lin[:, 1], c=y_lin, alpha=0.5)
plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
            facecolors="none", edgecolors="black", s=150, label="support vectors")
plt.legend()
plt.show()
```

## 2. C sweep and support vector count

```python
X_overlap, y_overlap = make_blobs(n_samples=200, centers=2, cluster_std=3.0, random_state=0)
for C in [0.01, 0.1, 1, 10, 100]:
    m = SVC(kernel="linear", C=C).fit(X_overlap, y_overlap)
    print(C, m.score(X_overlap, y_overlap), len(m.support_vectors_))
```

Larger `C` (penalizing margin violations more heavily) typically shrinks the
margin, which in turn reduces the number of points close enough to the
boundary to count as support vectors; smaller `C` allows a wider margin,
pulling in more points as support vectors since more of them fall within or
near it.

## 3. Linear vs RBF kernel on non-separable data

```python
X_circ, y_circ = make_circles(n_samples=200, noise=0.1, factor=0.4, random_state=0)

linear_model = SVC(kernel="linear").fit(X_circ, y_circ)
rbf_model = SVC(kernel="rbf").fit(X_circ, y_circ)

print("linear:", linear_model.score(X_circ, y_circ))   # poor, near 50%
print("rbf:", rbf_model.score(X_circ, y_circ))          # high, e.g. >95%
```

The linear kernel can only draw a straight line, which fundamentally cannot
separate a circle from a surrounding ring; the RBF kernel implicitly
projects into a space where this becomes linearly separable, recovering
high accuracy — a direct, hands-on demonstration of the kernel trick's
value.

## 4. Gamma sweep and overfitting

```python
Xc_train, Xc_test, yc_train, yc_test = train_test_split(X_circ, y_circ, test_size=0.3, random_state=0)

for gamma in [0.1, 1, 10, 100]:
    m = SVC(kernel="rbf", gamma=gamma).fit(Xc_train, yc_train)
    print(gamma, "train:", m.score(Xc_train, yc_train), "test:", m.score(Xc_test, yc_test))
```

At very high `gamma`, training accuracy stays near-perfect while test
accuracy can start dropping — each point's influence becomes so localized
that the boundary starts hugging individual training points rather than the
true circular pattern, the RBF-kernel equivalent of KNN's small-`k`
overfitting (Lesson 025).

## 5. SVM vs KNN on circles

```python
from sklearn.neighbors import KNeighborsClassifier
import time

knn = KNeighborsClassifier(n_neighbors=5).fit(Xc_train, yc_train)
print("knn test accuracy:", knn.score(Xc_test, yc_test))

best_rbf = SVC(kernel="rbf", gamma=1).fit(Xc_train, yc_train)
print("rbf svm test accuracy:", best_rbf.score(Xc_test, yc_test))
```

Both typically perform well on this cleanly-structured circular pattern;
KNN's prediction cost scales with training set size (Lesson 025) while a
fitted SVM's prediction cost scales with the number of support vectors,
often far fewer than the full training set — SVM prediction can be faster
at scale despite (typically) slower training.

## 6. Feature scaling for SVM

```python
rng = np.random.default_rng(0)
n = 300
feature_a = rng.uniform(0, 1, n)
feature_b = rng.uniform(0, 10000, n)
y2 = (feature_a > 0.5).astype(int)
X2 = np.column_stack([feature_a, feature_b])

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=0)

unscaled = SVC(kernel="rbf").fit(X2_train, y2_train)
print("unscaled:", unscaled.score(X2_test, y2_test))

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X2_train)
scaled = SVC(kernel="rbf").fit(scaler.transform(X2_train), y2_train)
print("scaled:", scaled.score(scaler.transform(X2_test), y2_test))
```

Unscaled, `feature_b`'s huge range dominates the RBF kernel's distance
calculation (`||x-x'||^2`) even though it's irrelevant noise, hurting
accuracy; after standardization both features contribute proportionally and
accuracy recovers — the same underlying reason KNN needed scaling
(Lesson 025), since both algorithms fundamentally rely on distance/dot-
product geometry.
