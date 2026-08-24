# 01 — Questions

Use this dataset for all questions (stdlib random, no extra files needed):

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(123)
n = 500

df = pd.DataFrame({
    "student_id": np.arange(1, n + 1),
    "hours_studied": rng.uniform(0, 10, n).round(1),
    "attendance_pct": rng.uniform(50, 100, n).round(1),
    "passed_prior_course": rng.choice([True, False], n, p=[0.7, 0.3]),
})

# exam_score depends on hours studied, attendance, and prior pass, plus noise
df["exam_score"] = (
    30
    + 4.5 * df["hours_studied"]
    + 0.2 * df["attendance_pct"]
    + 5 * df["passed_prior_course"]
    + rng.normal(0, 6, n)
).clip(0, 100).round(1)

# inject some missing attendance values
missing_idx = rng.choice(n, size=15, replace=False)
df.loc[missing_idx, "attendance_pct"] = np.nan
```

## Python fundamentals

1. Write a function `letter_grade(score)` that returns `"A"` for >=90, `"B"`
   for >=80, `"C"` for >=70, `"D"` for >=60, else `"F"`. Apply it to create a
   `grade` column (no explicit Python `for` loop over rows — use `.apply` or a
   vectorized approach).

## NumPy

2. Convert `df["exam_score"]` to a NumPy array. Without using `.mean()`
   directly, compute the mean using `.sum()` and `len()`, and confirm it
   matches `.mean()`.

3. Using boolean indexing on the NumPy array from Q2, count how many scores
   are within one standard deviation of the mean.

## Pandas

4. Handle the missing `attendance_pct` values: fill them with the column's
   median (justify in a comment why median might be preferred over mean here).

5. Using `groupby`, compute the average `exam_score` for students who passed
   their prior course vs those who didn't. Is the difference in the direction
   you'd expect given how the data was generated?

6. Create a new column `study_bucket` that bins `hours_studied` into
   `"low"` (<3.3), `"medium"` (3.3–6.6), `"high"` (>6.6) using
   `pd.cut`. Show the average `exam_score` per bucket.

## Visualization

7. Plot a scatter of `hours_studied` vs `exam_score`, colored by
   `passed_prior_course`.

8. Plot a heatmap of the correlation matrix of the numeric columns
   (`hours_studied`, `attendance_pct`, `exam_score`).

## Probability & statistics

9. Treat `passed_prior_course` as a Bernoulli random variable. From the data,
   estimate `P(passed_prior_course)`. Then estimate
   `P(grade == "A" | passed_prior_course == True)` and
   `P(grade == "A" | passed_prior_course == False)` directly from the data
   (no formula, just filtering + counting). Does prior success seem to raise
   the chance of an A?

10. Compute the Pearson correlation between `hours_studied` and `exam_score`.
    Then compute the z-score of `hours_studied` for the single student who
    scored the highest `exam_score`. Was that student unusually diligent, or
    could their score be mostly explained by attendance/prior course instead?
