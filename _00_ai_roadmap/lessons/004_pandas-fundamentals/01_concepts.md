# 01 — Concepts: Pandas Fundamentals

## Series and DataFrame

A `Series` is a 1D labeled array (like a NumPy array with an index attached).
A `DataFrame` is a 2D table: a dict of Series sharing the same index, each column
a `Series`.

```python
import pandas as pd

s = pd.Series([10, 20, 30], index=["a", "b", "c"])

df = pd.DataFrame({
    "name": ["Alice", "Bob", "Carol"],
    "age": [30, 25, 35],
    "city": ["NY", "LA", "NY"],
})
```

## Loading data

```python
df = pd.read_csv("data.csv")
df.head()          # first 5 rows
df.info()          # dtypes + non-null counts
df.describe()      # summary stats for numeric columns
df.shape            # (rows, cols)
```

## Selecting data

```python
df["age"]                     # a column -> Series
df[["name", "age"]]           # multiple columns -> DataFrame
df.loc[0]                     # row by label
df.iloc[0]                    # row by position
df.loc[df["age"] > 28]        # boolean filter (rows where condition is True)
df.loc[df["city"] == "NY", "name"]   # filter rows, select one column
```

`.loc` is label-based, `.iloc` is purely positional — mixing them up is one of the
most common pandas bugs.

## Adding / transforming columns

```python
df["age_in_5y"] = df["age"] + 5
df["is_adult"] = df["age"] >= 18
df["city_lower"] = df["city"].str.lower()
```

## Handling missing data

```python
df.isna().sum()            # count of missing values per column
df.dropna()                 # drop any row with a missing value
df.fillna(0)                 # replace missing with a constant
df["age"].fillna(df["age"].mean())   # replace with column mean
```

Deciding *how* to handle missing data (drop vs. impute, and with what) is itself a
modeling decision — revisit it once you know which model you're feeding the data
into.

## `groupby` — split-apply-combine

```python
df.groupby("city")["age"].mean()          # average age per city
df.groupby("city").agg(
    avg_age=("age", "mean"),
    count=("name", "count"),
)
```

`groupby` splits the DataFrame into groups by key, applies a function to each
group, then combines the results — the same three-step pattern behind almost
every aggregate report you'll ever compute.

## Merging / joining

```python
orders = pd.DataFrame({"order_id": [1, 2], "customer_id": [1, 2]})
customers = pd.DataFrame({"customer_id": [1, 2], "name": ["Alice", "Bob"]})

pd.merge(orders, customers, on="customer_id", how="left")
```

`how` works like SQL joins: `"inner"`, `"left"`, `"right"`, `"outer"`.

## Sorting and value counts

```python
df.sort_values("age", ascending=False)
df["city"].value_counts()      # frequency of each unique value
```
