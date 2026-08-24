# 02 — Practicals: Bagging & Random Forests

```python
from sklearn.datasets import make_classification
X, y = make_classification(
    n_samples=500, n_features=15, n_informative=5, n_redundant=5,
    random_state=0,
)
```

1. Fit a single `DecisionTreeClassifier` (no depth limit) and a
   `RandomForestClassifier(n_estimators=200)` on an 80/20 split. Compare
   test accuracy — how much does the forest improve on the single tree?

2. Fit the Random Forest with `oob_score=True`. Compare the OOB score to the
   held-out test accuracy from Q1 — are they close? Explain why OOB score is
   a legitimate "free" validation estimate (relate to how bootstrap sampling
   works).

3. Sweep `n_estimators` over `[1, 5, 20, 50, 100, 300]` and plot test
   accuracy vs number of trees. Does accuracy keep improving indefinitely,
   or plateau? At roughly what point are you getting diminishing returns?

4. Compare a `BaggingClassifier` (no feature subsetting) against a
   `RandomForestClassifier` (both with `n_estimators=200`), same data.
   Which performs better here? (Results can go either way depending on the
   dataset — report what you actually observe and reason about why, rather
   than assuming Random Forest always wins.)

5. Print feature importances from the Random Forest and compare to a single
   decision tree's importances (from Lesson 026's approach) on the same
   data. Which set of importances would you trust more, and why?

6. Introduce label noise: flip 10% of `y_train`'s labels randomly. Refit
   both a single tree and the Random Forest. Which degrades less in test
   accuracy? Explain using the "averaging cancels uncorrelated errors"
   argument from `01_concepts.md`.
