# 01 — Concepts: Decision Trees

## The model

A tree of if/else questions on feature values ("is `age < 30`?"), ending in
leaf nodes that predict a class (classification) or value (regression). Fully
interpretable — you can read off the exact decision path for any prediction,
unlike most other models in this curriculum.

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree

model = DecisionTreeClassifier(max_depth=3)
model.fit(X_train, y_train)
plot_tree(model, feature_names=feature_names, filled=True)
```

## How splits are chosen: impurity measures

At each node, the tree picks the feature and threshold that best separates
classes, measured by how much it reduces **impurity**:

- **Gini impurity**: `1 - Σ p_i^2` — probability of misclassifying a
  randomly picked point if you labeled it randomly according to the node's
  class proportions. 0 = pure node (all one class), max at equal class
  proportions.
- **Entropy** (Lesson 016): `-Σ p_i * log(p_i)` — same intuition, information-
  theoretic framing. **Information gain** = parent entropy minus the
  weighted average entropy of the children after a split; the tree picks the
  split that maximizes information gain (equivalently, minimizes weighted
  child impurity).

Both measures usually produce similar trees in practice; Gini is
`sklearn`'s default (`criterion="gini"`) and slightly cheaper to compute
(no logarithm).

## Growing the tree (greedy, not globally optimal)

At each node, the tree greedily picks the *locally* best split — it doesn't
look ahead to see if a worse split now enables better splits later. This
greedy approach is fast but not guaranteed to find the globally best tree;
in practice it works well enough, and ensembles (Lesson 027) compensate for
individual trees' greedy limitations.

## Overfitting: trees are extremely prone to it

An unconstrained tree can grow until every leaf is perfectly pure (often one
sample per leaf) — 100% training accuracy, terrible generalization. This is
Lesson 017's high-variance regime taken to an extreme. Control it via
hyperparameters:

- `max_depth`: limits how many splits deep the tree can go.
- `min_samples_split` / `min_samples_leaf`: requires a minimum number of
  samples before splitting / at a leaf — prevents splits based on just 1-2
  points.
- `max_features`: limits how many features are considered at each split
  (also used deliberately in Random Forests, Lesson 027, for a different
  reason — decorrelating trees).
- **Pruning** (`ccp_alpha` in sklearn — cost-complexity pruning): grow the
  full tree, then remove branches that don't improve validation performance
  enough to justify their added complexity.

Tune these via cross-validation, same as any hyperparameter.

## Feature importance

Trees naturally rank features by how much they reduce impurity across all
their splits (weighted by how many samples pass through each split) —
`model.feature_importances_`. Useful for a quick "what matters" check, though
correlated features can split credit between them in a way that
underrepresents each individually (similar caution to Lesson 021's
multicollinearity issue with linear regression coefficients).

## What trees are good and bad at

**Good**: capturing non-linear relationships and feature interactions
automatically (no manual polynomial/interaction terms needed, unlike Lessons
020–022); handling mixed numeric/categorical data without scaling (splits
are threshold-based, so feature scale doesn't matter — unlike KNN,
Lesson 025); full interpretability for shallow trees.

**Bad**: high variance/instability (small data changes can produce quite
different trees — motivates ensembles, Lesson 027); can't extrapolate beyond
the range of training data (a leaf's prediction is a fixed value/majority
class, regardless of how far a new point's features fall outside what was
seen); biased toward features with many possible split points (worth
knowing, less of a practical concern with modern implementations).
