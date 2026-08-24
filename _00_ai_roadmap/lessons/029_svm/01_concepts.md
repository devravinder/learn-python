# 01 — Concepts: Support Vector Machines

## The core idea: maximum margin

For linearly separable classes, there are infinitely many hyperplanes that
separate them perfectly. SVM picks the one that maximizes the **margin** —
the distance to the nearest point of *either* class. Those nearest points
are the **support vectors** — they're the only points that actually
determine the boundary; every other point could move around (without
crossing into the margin) and the boundary wouldn't change.

```python
from sklearn.svm import SVC
model = SVC(kernel="linear")
model.fit(X_train, y_train)
print(model.support_vectors_)
```

## Why maximum margin, specifically

Intuitively, a boundary that's as far as possible from both classes'
nearest points is more robust to noise in new data — a point similar to a
training example is less likely to fall on the wrong side. This is a
different philosophy from logistic regression's "maximize likelihood of the
observed labels" — SVM asks a geometric question instead of a probabilistic
one.

## Soft margin: allowing some violations

Real data usually isn't perfectly separable. The **soft margin** SVM allows
some points to violate the margin (or even be misclassified), controlled by
`C`:

```
minimize: (1/2)||w||^2 + C * Σ (margin violations)
```

- **Large `C`**: heavily penalizes violations → narrow margin, fits training
  data closely → higher variance (can overfit).
- **Small `C`**: tolerates more violations → wider margin, simpler boundary
  → higher bias (can underfit).

This is Lesson 017's bias-variance tradeoff again, controlled by `C` instead
of tree depth or polynomial degree.

## The kernel trick

For data that isn't linearly separable, SVM can implicitly map it into a
higher-dimensional space where it *is* separable — without ever explicitly
computing the transformed coordinates. This works because the SVM's
optimization only ever needs **dot products** between points (Lesson 010),
and a **kernel function** `K(x, x')` can compute the dot product *as if* the
points had been mapped to a higher-dimensional space, at the original
space's computational cost.

```python
model_rbf = SVC(kernel="rbf", gamma="scale")   # radial basis function kernel
model_poly = SVC(kernel="poly", degree=3)
```

- **RBF (Gaussian) kernel**: `K(x, x') = exp(-γ||x-x'||^2)` — effectively an
  infinite-dimensional feature space; the default choice for nonlinear data
  when you don't know the right explicit transformation.
- **Polynomial kernel**: equivalent to adding polynomial features
  (Lesson 022) without explicitly constructing them.
- `gamma` (RBF): controls how far a single training point's influence
  reaches — high `gamma` → influence very local (can overfit, similar to
  KNN's small-`k` regime); low `gamma` → influence very broad (can
  underfit).

## SVM for regression (SVR) — brief mention

The same margin idea generalizes to regression: fit a function such that
most points fall within a margin (`epsilon`) of the predicted line/curve,
penalizing only points that fall outside it. Less commonly used than
classification SVM in practice.

## When to reach for SVM vs trees/boosting

SVMs work well on smaller-to-medium datasets, especially with a clear margin
and moderate dimensionality, and were the default "strong nonlinear
classifier" choice before Gradient Boosting (Lesson 028) and neural networks
became dominant. They scale poorly to very large datasets (training
complexity is worse than linear in `n` for most kernels) and, like KNN,
require feature scaling (distances/dot products again). For most modern
tabular problems, Gradient Boosting or Random Forests are the more common
default; SVMs still see use in smaller, high-dimensional problems (e.g.
some bioinformatics/text classification tasks with strong margins).
