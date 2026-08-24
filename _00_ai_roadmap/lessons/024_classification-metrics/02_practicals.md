# 02 — Practicals: Classification Metrics in Practice

Build an imbalanced dataset:

```python
import numpy as np
from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=1000, n_features=10, n_informative=5,
    weights=[0.9, 0.1], random_state=0,
)
```

1. Split with `train_test_split(..., stratify=y)`. Fit
   `LogisticRegression`. Print `classification_report` and the confusion
   matrix.

2. Plot the ROC curve and report AUC. Then plot the Precision-Recall curve
   (`sklearn.metrics.precision_recall_curve` + matplotlib) for the same
   model. Which curve makes the class imbalance's effect on precision more
   visually obvious?

3. Sweep thresholds from 0.1 to 0.9 and find the threshold that maximizes
   F1 score. Compare precision/recall at that threshold vs the default 0.5
   threshold.

4. Suppose false negatives (missing the rare class) are 5x more costly than
   false positives in your application. Pick a threshold that reflects this
   priority (favor recall) and justify your choice with the numbers from
   Q3's sweep.

5. Refit without `stratify=y` in the split (try a few different
   `random_state` values) and observe how much the reported metrics vary
   run-to-run compared to the stratified version — quantify the instability
   a non-stratified split can introduce on imbalanced data.
