# 01 — Concepts: K-Nearest Neighbors

## The algorithm

To classify a new point: find the `k` closest points in the training data
(by some distance metric, usually Euclidean, Lesson 010), and predict the
majority class among them (for regression: their average value). There's no
real "training" step — KNN just memorizes the training data and does the
work at prediction time (a **lazy learner**, unlike every other model in
this module which fits parameters upfront).

```python
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)   # just stores the data
preds = model.predict(X_test)
```

## Choosing `k` — a direct bias-variance dial

- **Small `k`** (e.g. 1): the decision boundary follows every quirk of the
  training data closely → **low bias, high variance** (overfitting; very
  sensitive to noise/outliers in individual points).
- **Large `k`**: predictions average over more neighbors, smoothing the
  boundary → **high bias, low variance** (can underfit; `k = n` just
  predicts the overall majority class everywhere, ignoring `x` entirely).

Tune `k` via cross-validation (Lesson 017), same as any other
hyperparameter.

## Feature scaling is not optional

KNN's entire notion of "nearby" comes from distance calculations — an
unscaled feature with a huge numeric range (e.g. income in dollars) will
dominate the distance over a feature with a small range (e.g. age),
regardless of which one is actually more predictive. **Always standardize
features before KNN** (Lesson 008).

## Distance metrics

- **Euclidean** (`L2`, Lesson 010): default, straight-line distance.
- **Manhattan** (`L1`): sum of absolute differences — sometimes more robust
  when features aren't naturally continuous/geometric.
- **Cosine distance** (`1 - cosine similarity`, Lesson 010): common for text/
  embedding data where direction matters more than magnitude.

```python
model = KNeighborsClassifier(n_neighbors=5, metric="manhattan")
```

## The curse of dimensionality strikes here directly

Lesson 019 showed distances become less meaningful in high dimensions — KNN
is the algorithm most directly hurt by this, since "nearest neighbor" is its
entire mechanism. In high-dimensional feature spaces (many irrelevant
features, or raw pixels/text without embeddings), KNN often performs poorly
compared to models that don't rely purely on raw distance.

## Computational cost

Naive KNN prediction requires computing distance to *every* training point —
`O(n)` per prediction, which gets slow for large `n`. Real implementations
use spatial data structures (KD-trees, ball trees — `sklearn`'s default
`algorithm='auto'` picks one automatically) to speed this up, though even
those degrade toward brute-force in high dimensions (yet another curse-of-
dimensionality symptom).

## KNN for regression

`KNeighborsRegressor` predicts the **average** (or distance-weighted average)
of the `k` nearest neighbors' target values instead of a majority vote —
same core mechanism, continuous output.
