# 01 — Concepts: Classification Metrics in Practice

## `sklearn`'s metric toolkit

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
)

preds = model.predict(X_test)
probs = model.predict_proba(X_test)[:, 1]

print(confusion_matrix(y_test, preds))
print(classification_report(y_test, preds))
print(roc_auc_score(y_test, probs))
```

`classification_report` prints precision/recall/F1 per class plus
macro/weighted averages (Lesson 018) in one call — your default first check
after training any classifier.

## Threshold tuning

`model.predict()` uses a fixed 0.5 threshold on `predict_proba`. That default
is rarely the right operating point for an imbalanced or asymmetric-cost
problem (Lesson 018). Sweep thresholds explicitly:

```python
import numpy as np
thresholds = np.linspace(0.05, 0.95, 19)
for t in thresholds:
    preds_t = (probs >= t).astype(int)
    p = precision_score(y_test, preds_t, zero_division=0)
    r = recall_score(y_test, preds_t, zero_division=0)
    print(t, p, r)
```

Pick the threshold that best matches your problem's real precision/recall
priority (Lesson 018) — not necessarily 0.5.

## Plotting the ROC curve

```python
import matplotlib.pyplot as plt

fpr, tpr, thresholds = roc_curve(y_test, probs)
plt.plot(fpr, tpr, label=f"AUC={roc_auc_score(y_test, probs):.2f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")  # random-guess baseline
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()
```

The diagonal line represents a random classifier (AUC=0.5) — your curve
should bow toward the top-left corner; the further it bows, the higher the
AUC.

## Precision-Recall curve (better than ROC for rare positives)

When the positive class is rare (Lesson 018's disease/fraud examples), ROC
curves can look deceptively good because the False Positive *Rate*
denominator (all negatives) is huge, diluting the effect of false positives.
The **Precision-Recall curve** (`sklearn.metrics.precision_recall_curve`)
is more informative in that regime — it directly shows the
precision/recall tradeoff without a large-negative-class denominator masking
problems.

## Stratified train/test splits

For classification, always use `stratify=y` in `train_test_split` (or
`StratifiedKFold` for cross-validation) — this ensures the class balance in
each split matches the overall dataset, preventing a train/test split from
accidentally under/over-representing the rare class, which would give
noisy, unreliable metric estimates.

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=0
)
```
