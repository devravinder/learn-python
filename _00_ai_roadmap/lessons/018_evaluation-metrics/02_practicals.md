# 02 — Practicals: Evaluation Metrics

## Regression

1. Given `y_true = np.array([3, -0.5, 2, 7])` and
   `y_pred = np.array([2.5, 0.0, 2, 8])`, compute MAE, MSE, RMSE, and R² from
   scratch (no `sklearn.metrics`).

2. Add one large outlier to both arrays (`y_true` append `100`, `y_pred`
   append `50`) and recompute MAE and RMSE. Which metric moved more in
   relative terms? Why?

## Classification

Use this imbalanced dataset: 950 negatives, 50 positives (a rare-event
detector, e.g. fraud or disease). A lazy classifier always predicts negative;
a real classifier catches 40 of the 50 positives but also flags 30 false
positives.

3. Build the confusion matrices for both classifiers (lazy vs real). Compute
   accuracy, precision, recall, and F1 for each. Which metric makes the lazy
   classifier look good despite being useless? Which metrics expose it?

4. Given predicted probabilities and true labels:
   ```python
   y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0])
   y_prob = np.array([0.9, 0.4, 0.6, 0.3, 0.2, 0.7, 0.8, 0.1])
   ```
   Compute precision and recall at thresholds 0.3, 0.5, and 0.7. Confirm
   raising the threshold increases precision and decreases (or holds)
   recall.

5. Implement AUC from scratch via the "probability that a random positive is
   ranked higher than a random negative" definition: for every
   (positive, negative) pair, check if the positive's predicted probability
   is higher; AUC is the fraction of pairs where that holds. Compute it for
   the data in Q4 and sanity-check against `sklearn.metrics.roc_auc_score`
   if you have it installed.

6. For a spam filter and a rare-disease detector, argue (2–3 sentences each)
   which single metric (precision, recall, F1, or accuracy) you'd optimize
   for as the primary metric, and why the other one's asymmetric cost
   justifies a different choice than the first.
