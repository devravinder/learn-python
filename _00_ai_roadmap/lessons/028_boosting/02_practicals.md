# 02 — Practicals: Boosting

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=1000, n_features=20, n_informative=8, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

1. Fit `AdaBoostClassifier(n_estimators=100)` and
   `GradientBoostingClassifier(n_estimators=100)`, and compare their test
   accuracy to Lesson 027's `RandomForestClassifier(n_estimators=100)` on
   this same data.

2. For `GradientBoostingClassifier`, sweep `learning_rate` over
   `[0.01, 0.05, 0.1, 0.5, 1.0]` with `n_estimators=100` fixed. Plot test
   accuracy vs learning rate. Does a very high learning rate hurt?

3. Fix `learning_rate=0.05` and sweep `n_estimators` over
   `[10, 50, 100, 300, 500]`. Plot both train and test accuracy. At what
   point (if any) does test accuracy start declining while train accuracy
   keeps rising — the overfitting signature from Lesson 017?

4. If you have `xgboost` installed: fit `xgb.XGBClassifier` with an
   `eval_set` and `early_stopping_rounds=10`. Report the number of trees it
   actually used (`model.best_iteration`) vs the `n_estimators` you
   requested. If not installed, implement the concept manually: train
   `GradientBoostingClassifier` with `n_estimators=500`, but manually track
   validation accuracy every 20 trees (via `staged_predict`) and report the
   point where it stops improving.

5. Compare training wall-clock time (`time.time()`) between
   `RandomForestClassifier(n_estimators=200)` and
   `GradientBoostingClassifier(n_estimators=200)` on the same data. Which is
   faster to train, and why (relate to "parallelizable" vs "sequential" from
   `01_concepts.md`)?

6. Print feature importances from the best Gradient Boosting model and
   compare the top 3 features to the Random Forest's top 3 from Lesson 027's
   approach — do the two ensemble types agree on what matters most?
