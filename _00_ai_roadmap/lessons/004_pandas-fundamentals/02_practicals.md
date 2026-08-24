# 02 — Practicals: Pandas Fundamentals

Use this small in-memory dataset for every exercise (no file needed):

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    "name":   ["Alice", "Bob", "Carol", "Dan", "Eve", "Frank"],
    "age":    [30, 25, 35, np.nan, 28, 40],
    "city":   ["NY", "LA", "NY", "SF", "LA", "SF"],
    "salary": [70000, 60000, 90000, 65000, 72000, 110000],
})
```

1. Print `df.info()` and `df.describe()`. Which column has missing data, and how
   many missing values?

2. Select just the `name` and `salary` columns for everyone older than 28
   (careful: one row has a missing age — decide and note what happens to it).

3. Fill the missing `age` with the column's mean, then add a column `age_group`
   that is `"young"` if age < 30 else `"senior"`.

4. Compute the average `salary` per `city`, sorted descending.

5. Add a column `salary_rank` giving each row's rank of `salary` within its
   `city` group (highest salary = rank 1). Hint: look up `groupby(...).rank()`.

6. Create a second DataFrame:
   ```python
   city_info = pd.DataFrame({
       "city": ["NY", "LA", "SF"],
       "cost_of_living_index": [187, 173, 200],
   })
   ```
   Merge it onto `df` so every row also has its city's `cost_of_living_index`.

7. Using the merged result, add a column `adjusted_salary` = `salary / cost_of_living_index * 100`,
   then find the name of the person with the highest `adjusted_salary`.
