# 03 — Solutions: Evaluation Metrics

## 1. Regression metrics from scratch

```python
import numpy as np

y_true = np.array([3, -0.5, 2, 7])
y_pred = np.array([2.5, 0.0, 2, 8])

mae = np.mean(np.abs(y_true - y_pred))
mse = np.mean((y_true - y_pred) ** 2)
rmse = np.sqrt(mse)

ss_res = np.sum((y_true - y_pred) ** 2)
ss_tot = np.sum((y_true - y_true.mean()) ** 2)
r2 = 1 - ss_res / ss_tot

print(mae, mse, rmse, r2)
```

## 2. Outlier sensitivity

```python
y_true2 = np.append(y_true, 100)
y_pred2 = np.append(y_pred, 50)

mae2 = np.mean(np.abs(y_true2 - y_pred2))
rmse2 = np.sqrt(np.mean((y_true2 - y_pred2) ** 2))
print(mae2, rmse2)
```

RMSE increases proportionally *much* more than MAE relative to their
original values, because squaring a 50-unit error (2500) dominates the sum
far more than adding 50 linearly does — this is exactly why RMSE is
preferred when large errors are especially costly, and MAE when you don't
want one outlier to dominate the metric.

## 3. Confusion matrix on imbalanced data

```python
# Lazy classifier: always predicts negative
# TP=0, FP=0, FN=50, TN=950
lazy_acc = (0 + 950) / 1000            # 0.95
lazy_precision = 0 / (0 + 0 + 1e-9)     # undefined/0 (no positive predictions at all)
lazy_recall = 0 / 50                    # 0.0

# Real classifier: TP=40, FP=30, FN=10, TN=920
real_acc = (40 + 920) / 1000            # 0.96
real_precision = 40 / (40 + 30)         # 0.571
real_recall = 40 / (40 + 10)            # 0.8
real_f1 = 2 * real_precision * real_recall / (real_precision + real_recall)  # ~0.667

print(lazy_acc, real_acc)
print(real_precision, real_recall, real_f1)
```

**Accuracy makes the lazy classifier look deceptively good** (95%, barely
below the real classifier's 96%) despite catching zero real positive cases —
exactly the failure mode from Lesson 006's disease-testing example.
**Recall (0.0 vs 0.8) and F1 immediately expose it** as useless.

## 4. Precision/recall at different thresholds

```python
y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0])
y_prob = np.array([0.9, 0.4, 0.6, 0.3, 0.2, 0.7, 0.8, 0.1])

def precision_recall_at(threshold):
    y_pred = (y_prob >= threshold).astype(int)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    return precision, recall

for t in [0.3, 0.5, 0.7]:
    print(t, precision_recall_at(t))
```

As the threshold rises, fewer examples get predicted positive — recall can
only decrease or stay the same, while precision generally increases (fewer,
more confident positive predictions).

## 5. AUC from scratch via pairwise ranking

```python
def auc_from_scratch(y_true, y_prob):
    pos_scores = y_prob[y_true == 1]
    neg_scores = y_prob[y_true == 0]
    count = 0
    total = 0
    for p in pos_scores:
        for n in neg_scores:
            total += 1
            if p > n:
                count += 1
            elif p == n:
                count += 0.5
    return count / total

print(auc_from_scratch(y_true, y_prob))
```

This should match `sklearn.metrics.roc_auc_score(y_true, y_prob)` exactly —
AUC's "probability a random positive outranks a random negative" definition
is mathematically equivalent to the area under the ROC curve.

## 6. Metric choice for spam vs disease detection

**Spam filter**: prioritize **precision**. A false positive (real email
marked spam) can mean a missed job offer or important message — an
expensive, visible failure — while a false negative (one spam email getting
through) is a minor annoyance. Optimize to minimize false positives even at
some recall cost.

**Rare-disease detector**: prioritize **recall**. A false negative (missed
real disease case) can be life-threatening, while a false positive (healthy
person flagged, then given a confirmatory follow-up test) is a comparatively
minor cost. Optimize to catch as many true cases as possible even if it
means more people get send for extra (cheaper, less risky) confirmation.
