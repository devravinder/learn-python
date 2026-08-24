# 01 — Concepts: Boosting

## The core idea: sequential correction, not parallel averaging

Bagging/Random Forests (Lesson 027) train trees **independently** and
average. Boosting trains trees **sequentially**: each new tree focuses on
the mistakes the ensemble-so-far is making, and gets added to the ensemble
with some weight. This targets **bias** (an individual "weak learner" — often
a shallow tree with just a few splits — is deliberately not very powerful on
its own; boosting is what makes the *ensemble* powerful), the opposite focus
from bagging's variance reduction.

## AdaBoost (Adaptive Boosting)

1. Start with equal weights on all training samples.
2. Train a weak learner (commonly a "stump" — a depth-1 tree).
3. Increase the weight of samples it got **wrong**, decrease weight of
   samples it got **right** — so the next weak learner is forced to focus
   more on the hard cases.
4. Repeat; combine all weak learners into a weighted vote (learners that
   performed better get more say).

```python
from sklearn.ensemble import AdaBoostClassifier
model = AdaBoostClassifier(n_estimators=100)
model.fit(X_train, y_train)
```

## Gradient Boosting — a more general framework

Instead of reweighting samples, Gradient Boosting fits each new tree to the
**residual errors** (or more precisely, the negative gradient of the loss
function with respect to the current predictions — hence "gradient"
boosting, directly connecting back to Lesson 015's gradient descent, but
descending through *function space* — adding trees — rather than through
parameter space):

```
F_0(x) = initial prediction (e.g. mean of y)
for m in 1..M:
    residuals = y - F_{m-1}(x)                    # (simplified: for MSE loss)
    tree_m = fit a tree to predict `residuals`
    F_m(x) = F_{m-1}(x) + learning_rate * tree_m(x)
```

Each tree is a small, incremental correction; `learning_rate` (also called
`shrinkage`) controls how much each tree's correction counts — lower values
need more trees but generalize better (same bias-variance tradeoff
philosophy as Lesson 015's learning rate, applied per-tree instead of
per-gradient-step).

```python
from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=3)
model.fit(X_train, y_train)
```

## XGBoost / LightGBM: production-grade gradient boosting

The same core idea, heavily engineered: built-in L1/L2 regularization on
tree weights (Lesson 022's regularization idea applied to tree ensembles),
efficient handling of missing values, parallelized tree construction, and
typically the strongest out-of-the-box performance on tabular data of
anything in this lesson.

```python
import xgboost as xgb
model = xgb.XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=4)
model.fit(X_train, y_train)
```

## Boosting's overfitting risk — different shape than a single tree's

Boosting *can* overfit if you use too many estimators with too high a
learning rate and too little regularization (each successive tree keeps
fitting residuals, eventually fitting noise). Mitigate with:
- **Early stopping**: monitor validation loss each round, stop adding trees
  once it stops improving (`eval_set` + `early_stopping_rounds` in XGBoost).
- Lower `learning_rate` + more `n_estimators` (slower but smoother
  learning, usually a better tradeoff than high learning rate + few
  estimators).
- Shallow trees (`max_depth` 3-6 typical) — each individual tree stays a
  genuinely "weak" learner.

## Random Forest vs Gradient Boosting: when to reach for which

| | Random Forest | Gradient Boosting / XGBoost |
|---|---|---|
| Training | Parallelizable (independent trees) | Sequential (each tree depends on the last) |
| Tuning effort | Low — good results with defaults | Higher — learning rate, depth, n_estimators all interact |
| Typical accuracy | Strong | Usually edges out RF when well-tuned |
| Overfitting behavior | More trees rarely hurts | Too many/aggressive trees can overfit |
| Good default when... | You want a fast, low-effort strong baseline | You have time to tune and want the best tabular performance |
