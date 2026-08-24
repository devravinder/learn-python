# 03 — Solutions: Data Visualization

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

rng = np.random.default_rng(7)
n = 200

df = pd.DataFrame({
    "age": rng.integers(18, 70, n),
    "city": rng.choice(["NY", "LA", "SF"], n),
    "salary": rng.normal(70000, 15000, n).round(-2),
})
df["salary"] += (df["age"] - 18) * 500
```

## 1. Age histogram

```python
fig, ax = plt.subplots()
ax.hist(df["age"], bins=15)
ax.set_xlabel("Age")
ax.set_ylabel("Count")
ax.set_title("Age distribution")
plt.show()
```

## 2. Salary histogram + KDE

```python
sns.histplot(data=df, x="salary", kde=True)
```

With this generator, salary is built from a normal distribution plus a linear
age term, so it should look roughly unimodal and close to normal, maybe
slightly right-skewed depending on the random seed.

## 3. Scatter, colored by city

```python
sns.scatterplot(data=df, x="age", y="salary", hue="city")
```

Because `salary` was constructed as `normal + (age - 18) * 500`, there should be
a visible upward trend of salary with age, with city adding noise/offset rather
than changing the trend's direction.

## 4. Boxplot by city

```python
sns.boxplot(data=df, x="city", y="salary")
```

Since `city` was assigned independently of `salary` in this synthetic dataset,
the three boxes should look similar in spread — any visible difference here is
sampling noise, which is itself a useful thing to learn to recognize.

## 5. Correlation heatmap

```python
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, vmin=-1, vmax=1, cmap="coolwarm")
```

Expect a clearly positive correlation between `age` and `salary` (by
construction), close to 0 correlation involving any column not causally linked
to another.

## 6. Side-by-side subplots

```python
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].hist(df["age"], bins=15)
axes[0].set_title("Age")
axes[1].hist(df["salary"], bins=15)
axes[1].set_title("Salary")
plt.tight_layout()
plt.show()
```

## 7. Corrupted data diagnosis

```python
bad = df.copy()
bad.loc[bad.sample(5, random_state=0).index, "salary"] = -1

sns.histplot(data=bad, x="salary", kde=True)
```

The corrupted rows show up as an isolated spike/tail near -1, far outside the
otherwise smooth bell shape — a histogram makes impossible values (negative
salary) visually obvious in a way that scanning a table of 200 rows would not.
The fix is **not** to just delete the outlier visually — go back to the data
source, understand why `-1` appears (often a sentinel for "missing" used
upstream), and treat it as missing data (drop or impute) rather than as a real
low salary.
