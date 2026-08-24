# 03 — Solutions: Descriptive Statistics

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(1)
df = pd.DataFrame({
    "hours_studied": rng.uniform(0, 10, 100),
})
df["exam_score"] = 40 + 5 * df["hours_studied"] + rng.normal(0, 8, 100)
df.loc[95:, "exam_score"] = [10, 12, 8, 15, 9]
```

## 1. Mean, median, std, IQR

```python
mean = df["exam_score"].mean()
median = df["exam_score"].median()
std = df["exam_score"].std()
q1, q3 = df["exam_score"].quantile([0.25, 0.75])
iqr = q3 - q1
print(mean, median, std, iqr)
```

The **mean is pulled down more than the median** by the 5 low outliers,
because the mean incorporates every value's exact magnitude while the median
only cares about rank order — five unusually low values shift the mean
directly but only shift the median if they change which value sits in the
middle.

## 2. Percentiles

```python
np.percentile(df["exam_score"], [25, 50, 75, 90])
```

## 3. Outlier detection via IQR rule

```python
q1, q3 = df["exam_score"].quantile([0.25, 0.75])
iqr = q3 - q1
lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr

outliers = df[(df["exam_score"] < lower) | (df["exam_score"] > upper)]
print(outliers.index)
```

This should catch some or all of the 5 injected rows (index 95-99) — whether
it catches *all five* depends on how far the random "normal" scores spread
the IQR; it's a heuristic, not a guarantee, which is exactly why you look at
the actual rows flagged rather than trusting the rule blindly.

## 4. Correlation two ways

```python
# (a) built-in
corr_np = np.corrcoef(df["hours_studied"], df["exam_score"])[0, 1]

# (b) manual
x, y = df["hours_studied"], df["exam_score"]
cov = ((x - x.mean()) * (y - y.mean())).mean()
corr_manual = cov / (x.std(ddof=0) * y.std(ddof=0))

print(corr_np, corr_manual)   # match (use ddof=0 to match np.corrcoef's population std)
```

## 5. Standardization

```python
z = (df - df.mean()) / df.std()
print(z.mean(), z.std())   # ~0, ~1 for both columns
```

## 6. Correlation without outliers

```python
clean = df.drop(index=range(95, 100))
corr_clean = np.corrcoef(clean["hours_studied"], clean["exam_score"])[0, 1]
print(corr_clean, "vs original", corr_np)
```

The correlation typically moves noticeably closer to the "true" underlying
relationship's strength once the 5 outliers (which don't follow the
`hours_studied` -> `exam_score` linear trend at all) are removed — a concrete
demonstration of why a single Pearson correlation number should never be
trusted without also looking at a scatter plot for points that don't fit the
pattern.
