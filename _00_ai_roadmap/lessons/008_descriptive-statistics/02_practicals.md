# 02 — Practicals: Descriptive Statistics

Use this dataset:

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(1)
df = pd.DataFrame({
    "hours_studied": rng.uniform(0, 10, 100),
})
df["exam_score"] = 40 + 5 * df["hours_studied"] + rng.normal(0, 8, 100)
df.loc[95:, "exam_score"] = [10, 12, 8, 15, 9]   # 5 low-scoring outliers
```

1. Compute mean, median, standard deviation, and IQR of `exam_score`. Given the
   5 injected outliers, which of mean/median moved more? Why?

2. Compute the 25th, 50th, 75th, and 90th percentiles of `exam_score` using
   `np.percentile`.

3. Using the `1.5*IQR` rule, identify which rows (if any) count as outliers in
   `exam_score`. Do they match the 5 rows you know were injected?

4. Compute the Pearson correlation between `hours_studied` and `exam_score`
   two ways: (a) with `np.corrcoef`, (b) manually from the covariance and
   standard deviation formulas. Confirm they match.

5. Standardize both columns (z-score) and confirm the standardized columns
   have mean ≈ 0 and std ≈ 1.

6. Remove the 5 outlier rows and recompute the correlation from Q4. How much
   does it change? What does that tell you about outliers' effect on
   correlation?
