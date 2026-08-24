# 01 — Concepts: Descriptive Statistics

## Central tendency

- **Mean**: `μ = (1/n) Σ xᵢ` — sensitive to outliers (one huge value drags it a lot).
- **Median**: the middle value when sorted — robust to outliers.
- **Mode**: the most frequent value — the only one of the three that makes
  sense for categorical data.

Rule of thumb: if mean and median differ a lot, the data is skewed (Lesson
005's order-value distribution was a real example: mean ~$194 vs median ~$48).

## Spread

- **Variance**: `σ² = (1/n) Σ (xᵢ - μ)²` — average squared deviation from the
  mean. Squaring makes it always positive and penalizes large deviations more.
- **Standard deviation**: `σ = √(σ²)` — same units as the original data (unlike
  variance, which is in squared units), so it's more interpretable.
- **Population vs sample variance**: dividing by `n` gives the population
  variance; dividing by `n-1` (Bessel's correction) gives an unbiased estimate
  of variance *from a sample* — `pandas`/`numpy`'s `.std()`/`.var()` default
  to `ddof=0` (population) in NumPy but `ddof=1` (sample) in Pandas. Always
  check which one a function is giving you.
- **Range**: `max - min` — simple, but entirely determined by the two most
  extreme points.
- **Interquartile range (IQR)**: `Q3 - Q1` (75th percentile minus 25th
  percentile) — robust spread measure, the basis of the boxplot's box, and a
  common way to define outliers: anything beyond `Q1 - 1.5*IQR` or
  `Q3 + 1.5*IQR`.

## Shape

- **Skewness**: measures asymmetry. Positive/right skew = long tail toward
  high values (income, order value); negative/left skew = long tail toward low
  values.
- **Kurtosis**: measures how heavy the tails are compared to a Normal
  distribution (more/fewer extreme outliers than "expected").

## Covariance and correlation

**Covariance** measures whether two variables move together:

```
Cov(X, Y) = (1/n) Σ (xᵢ - μx)(yᵢ - μy)
```

Positive covariance: they tend to move in the same direction. Negative: opposite
directions. Covariance's magnitude depends on the variables' units, which makes
it hard to compare across variable pairs — that's what correlation fixes.

**Pearson correlation** normalizes covariance by each variable's standard
deviation, giving a unitless value in `[-1, 1]`:

```
r = Cov(X, Y) / (σx * σy)
```

`r = 1`: perfect positive linear relationship. `r = -1`: perfect negative.
`r = 0`: no *linear* relationship (there could still be a strong non-linear
one — always plot the data, don't just trust the number; this is the core
lesson of "Anscombe's quartet," worth a quick search).

## Percentiles and quantiles

The `p`th percentile is the value below which `p`% of the data falls. Median =
50th percentile. `describe()` in pandas reports the 25th/50th/75th percentiles
by default — a quick shape summary without plotting anything.

## Standardization (z-score, revisited)

`z = (x - μ) / σ` rescales a variable to mean 0, std 1. Almost every classical
ML algorithm that relies on distances or gradients (KNN, SVM, gradient
descent-based models, neural networks) trains better/faster on standardized
features, because raw features on wildly different scales (e.g. "age" 0-100
vs "income" 0-1,000,000) otherwise dominate distance calculations or gradient
magnitudes.
