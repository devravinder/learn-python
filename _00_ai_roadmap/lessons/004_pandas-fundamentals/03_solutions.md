# 03 — Solutions: Pandas Fundamentals

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

## 1. Inspect

```python
df.info()
df.describe()
df.isna().sum()   # age: 1 missing
```

## 2. Filter + select

```python
df.loc[df["age"] > 28, ["name", "salary"]]
```

Dan's row has `age = NaN`; `NaN > 28` evaluates to `False`, so Dan is silently
excluded. This is a common, easy-to-miss bug source — always check `isna()`
before filtering on a column that has missing values.

## 3. Fill + derive column

```python
df["age"] = df["age"].fillna(df["age"].mean())
df["age_group"] = df["age"].apply(lambda a: "young" if a < 30 else "senior")
```

## 4. Average salary per city

```python
df.groupby("city")["salary"].mean().sort_values(ascending=False)
```

## 5. Rank within group

```python
df["salary_rank"] = df.groupby("city")["salary"].rank(ascending=False)
```

## 6. Merge

```python
city_info = pd.DataFrame({
    "city": ["NY", "LA", "SF"],
    "cost_of_living_index": [187, 173, 200],
})

merged = pd.merge(df, city_info, on="city", how="left")
```

## 7. Adjusted salary

```python
merged["adjusted_salary"] = merged["salary"] / merged["cost_of_living_index"] * 100
top_earner = merged.loc[merged["adjusted_salary"].idxmax(), "name"]
print(top_earner)
```

`idxmax()` returns the **index label** of the row with the maximum value, which
you then use with `.loc` to pull out the corresponding name — a very common
pandas idiom ("find the row where column X is highest, tell me column Y").
