# 01 — Concepts: Bagging & Random Forests

## The core idea: averaging reduces variance

If you have several models that each make somewhat-independent errors,
averaging their predictions cancels out a lot of that error — this is a
direct statistical fact (variance of a mean of `n` independent, identically
distributed variables is `1/n` times the variance of a single one). Decision
trees (Lesson 026) are high-variance, low-bias models — an ideal candidate
for this trick, since averaging attacks variance directly without needing to
fix bias.

## Bootstrap Aggregating (Bagging)

1. Create `n` **bootstrap samples**: randomly sample the training data *with
   replacement* to the same size as the original (so each sample sees ~63%
   of unique original points, some repeated, some left out — the left-out
   ~37% is the **out-of-bag (OOB)** set for that tree, usable as a free
   validation set).
2. Train one tree on each bootstrap sample, independently.
3. Predict by averaging (regression) or majority vote (classification)
   across all trees.

```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

model = BaggingClassifier(DecisionTreeClassifier(), n_estimators=100, oob_score=True)
model.fit(X_train, y_train)
print(model.oob_score_)   # free validation estimate, no held-out set needed
```

## Random Forests: bagging + extra decorrelation

Plain bagging's trees are still correlated with each other (they're all
grown on similar data, and a single dominant feature will show up as the
first split in most of them, making the trees more similar than the
bootstrap sampling alone would suggest). **Random Forests** add one more
randomization: at each split, only consider a random subset of features
(`max_features`, commonly `sqrt(n_features)` for classification), forcing
trees to sometimes use different, weaker features — deliberately
decorrelating the ensemble, which improves the variance-reduction benefit of
averaging.

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=200, max_features="sqrt", random_state=0)
model.fit(X_train, y_train)
```

## Why averaging many high-variance, low-bias models works so well

Each individual tree overfits its own bootstrap sample (Lesson 026's known
weakness). But since each tree overfits to *different* noise (different
bootstrap sample, different feature subsets), those errors are
largely uncorrelated and cancel out on average — leaving the shared signal
(the real pattern every tree partially captures) intact while variance drops
sharply. This is the single biggest reason Random Forests reliably
outperform a single decision tree with little tuning effort.

## Hyperparameters that matter

- `n_estimators`: more trees almost always helps (or at worst plateaus) —
  unlike other model complexity knobs, more trees doesn't cause overfitting
  by itself, though it does cost more compute.
- `max_depth`, `min_samples_leaf`: still relevant per-tree, though Random
  Forests are noticeably more forgiving of deep, unpruned trees than a
  single decision tree would be, since the ensemble averaging absorbs a lot
  of the individual overfitting.
- `max_features`: controls decorrelation strength; too low can hurt
  individual tree quality, too high approaches plain bagging.

## Feature importance in forests

Same idea as single trees (Lesson 026), averaged across all trees —
generally more stable/trustworthy than a single tree's importances, since
it's not dependent on one tree's particular (possibly unlucky) structure.

## When to prefer Random Forests over a single tree or linear model

Random Forests are a strong, low-effort default for tabular data: they
handle nonlinearity and feature interactions automatically, need minimal
feature scaling/preprocessing, and rarely require heavy hyperparameter
tuning to get decent results — though Gradient Boosting (Lesson 028) usually
edges them out on raw predictive performance when properly tuned, at the
cost of being more tuning-sensitive.
