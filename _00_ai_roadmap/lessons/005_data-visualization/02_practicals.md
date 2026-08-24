# 02 — Practicals: Data Visualization

Use this synthetic dataset for every exercise:

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(7)
n = 200

df = pd.DataFrame({
    "age": rng.integers(18, 70, n),
    "city": rng.choice(["NY", "LA", "SF"], n),
    "salary": rng.normal(70000, 15000, n).round(-2),
})
df["salary"] += (df["age"] - 18) * 500   # older -> generally higher salary
```

1. Plot a histogram of `age` with 15 bins. Add axis labels and a title.

2. Plot a histogram of `salary` with a KDE overlay (`sns.histplot(..., kde=True)`).
   Does it look roughly normal, skewed, or bimodal?

3. Make a scatter plot of `age` vs `salary`, colored by `city` (`hue="city"`).
   Does there appear to be a relationship between age and salary?

4. Make a boxplot of `salary` grouped by `city`. Which city has the widest
   salary spread?

5. Compute the correlation matrix of the numeric columns and plot it as an
   annotated heatmap.

6. Create a 1x2 figure (`plt.subplots(1, 2, figsize=(10, 4))`): left panel a
   histogram of `age`, right panel a histogram of `salary`, sharing one figure.

7. **Diagnosis exercise:** inject 5 corrupted rows with `salary = -1` into a
   copy of `df`. Replot the salary histogram from step 2 — describe how the
   plot makes the bad data visible, and what you'd do about it before training
   a model on this data.
