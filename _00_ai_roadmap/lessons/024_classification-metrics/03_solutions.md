# 03 — Solutions: Classification Metrics in Practice

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, precision_score, recall_score, f1_score,
)
import matplotlib.pyplot as plt

X, y = make_classification(
    n_samples=1000, n_features=10, n_informative=5,
    weights=[0.9, 0.1], random_state=0,
)
```

## 1. Fit and report

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=0
)
model = LogisticRegression().fit(X_train, y_train)
preds = model.predict(X_test)

print(confusion_matrix(y_test, preds))
print(classification_report(y_test, preds))
```

## 2. ROC vs Precision-Recall curve

```python
probs = model.predict_proba(X_test)[:, 1]

fpr, tpr, _ = roc_curve(y_test, probs)
plt.plot(fpr, tpr)
plt.title(f"ROC (AUC={roc_auc_score(y_test, probs):.2f})")
plt.show()

precision, recall, _ = precision_recall_curve(y_test, probs)
plt.plot(recall, precision)
plt.title("Precision-Recall")
plt.show()
```

With only ~10% positive class, the ROC curve often still looks quite good
(bowing toward the top-left) because the false positive *rate* denominator
(all the many negatives) dilutes the visual impact of false positives; the
Precision-Recall curve shows precision dropping more visibly as recall
increases, making the real difficulty of the imbalanced problem more
apparent.

## 3. Threshold sweep for best F1

```python
thresholds = np.linspace(0.1, 0.9, 17)
best_f1, best_t = -1, None
for t in thresholds:
    preds_t = (probs >= t).astype(int)
    f1 = f1_score(y_test, preds_t, zero_division=0)
    if f1 > best_f1:
        best_f1, best_t = f1, t

print("best threshold:", best_t, "F1:", best_f1)
preds_default = (probs >= 0.5).astype(int)
preds_best = (probs >= best_t).astype(int)
print("default 0.5:", precision_score(y_test, preds_default), recall_score(y_test, preds_default))
print(f"best {best_t}:", precision_score(y_test, preds_best), recall_score(y_test, preds_best))
```

The F1-optimal threshold is often below 0.5 on an imbalanced dataset like
this, since the model's predicted probabilities tend to be conservative
about the rare class — lowering the threshold catches more true positives
at some precision cost, often netting a better F1 than the default.

## 4. Recall-favoring threshold for asymmetric costs

```python
for t in thresholds:
    preds_t = (probs >= t).astype(int)
    p = precision_score(y_test, preds_t, zero_division=0)
    r = recall_score(y_test, preds_t, zero_division=0)
    print(round(t, 2), round(p, 3), round(r, 3))
```

With false negatives 5x costlier, pick a lower threshold from this printout
that pushes recall noticeably higher even at a real precision cost — e.g. if
`t=0.3` gives recall ~0.85 vs `t=0.5`'s recall ~0.65, the 5x cost asymmetry
easily justifies accepting more false positives to catch more true
positives. The right specific number depends on reading your own sweep's
printed values.

## 5. Instability without stratification

```python
for seed in range(5):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed)
    m = LogisticRegression().fit(Xtr, ytr)
    p = m.predict(Xte)
    print(seed, "positive rate in test:", yte.mean(), "recall:", recall_score(yte, p, zero_division=0))
```

Without `stratify=y`, the fraction of positives landing in the test set can
vary meaningfully run-to-run (especially with only ~100 total positives in a
1000-row dataset), and recall estimates swing correspondingly — a direct,
measurable illustration of why stratified splitting is the correct default
for imbalanced classification.
