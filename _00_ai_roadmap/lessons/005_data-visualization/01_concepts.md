# 01 — Concepts: Data Visualization

## Matplotlib basics

Matplotlib is the low-level plotting library everything else (including Seaborn
and Pandas' `.plot()`) is built on.

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot([1, 2, 3, 4], [1, 4, 9, 16])
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("y = x^2")
plt.show()
```

Prefer the explicit `fig, ax = plt.subplots()` form over bare `plt.plot(...)` —
it scales cleanly to multi-panel figures (`plt.subplots(2, 2)` gives a 2x2 grid
of `ax` objects).

## Choosing the right chart

| Question | Chart |
|---|---|
| How is one numeric column distributed? | histogram, KDE, boxplot |
| How do two numeric columns relate? | scatter plot |
| How does a value change over time? | line plot |
| How do groups compare on one metric? | bar chart, boxplot per group |
| How correlated are many numeric columns? | heatmap of the correlation matrix |

## Matplotlib essentials

```python
plt.hist(df["age"], bins=20)
plt.scatter(df["age"], df["salary"])
plt.plot(df["date"], df["value"])
plt.bar(df["city"], df["avg_salary"])
plt.boxplot(df["salary"])
```

## Seaborn — statistical plots with less code

Seaborn understands DataFrames directly (`data=df, x=..., y=...`) and adds
sensible defaults (confidence intervals, color palettes, grouping by a `hue`).

```python
import seaborn as sns

sns.histplot(data=df, x="age", kde=True)
sns.scatterplot(data=df, x="age", y="salary", hue="city")
sns.boxplot(data=df, x="city", y="salary")
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
```

`hue="city"` colors points/bars by category — the fastest way to add a third
dimension to a 2D plot.

## Correlation heatmaps

```python
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, vmin=-1, vmax=1, cmap="coolwarm")
```

Correlation ranges from -1 (perfectly inverse) to +1 (perfectly aligned); a
heatmap makes it fast to spot which features move together before you even
train a model.

## Small multiples (facets)

```python
g = sns.FacetGrid(df, col="city")
g.map(sns.histplot, "salary")
```

Facets repeat the same plot once per category — useful for comparing a
distribution across groups without cramming everything into one axis.

## Reading, not just making, plots

For every plot you produce, ask: what's the range of the axes, is the
distribution skewed, are there obvious outliers, and does anything look like a
data-entry error rather than a real pattern? EDA is about noticing what's
*wrong* with the data as much as what's interesting about it.
