# 03 — Solutions: Inferential Statistics & Hypothesis Testing

## 1. CLT visualization

```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
population = rng.normal(50, 10, 1_000_000)

sample_means = [rng.choice(population, 30).mean() for _ in range(1000)]

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].hist(population, bins=50)
axes[0].set_title("Population")
axes[1].hist(sample_means, bins=50)
axes[1].set_title("Sampling distribution of the mean (n=30)")
plt.show()
```

Both look roughly bell-shaped here (the population itself is Normal by
construction), but the sampling distribution is **much narrower** — its
spread is the population's std divided by `√30`, illustrating the standard
error formula directly. The CLT's real power shows even when the *population*
isn't Normal (try `rng.exponential(10, 1_000_000)` as the population instead
— the sampling distribution of the mean still becomes bell-shaped).

## 2. Confidence interval coverage

```python
def ci_95(sample):
    mean = sample.mean()
    se = sample.std(ddof=1) / np.sqrt(len(sample))
    return mean - 1.96 * se, mean + 1.96 * se

sample = rng.choice(population, 30)
lower, upper = ci_95(sample)
print(lower, upper, "contains 50:", lower <= 50 <= upper)

hits = 0
for _ in range(100):
    s = rng.choice(population, 30)
    lo, hi = ci_95(s)
    if lo <= 50 <= hi:
        hits += 1
print(f"{hits}/100 intervals contained the true mean")   # ~95/100
```

## 3. Paired t-test for model comparison

```python
from scipy import stats

rng = np.random.default_rng(1)
errors_a = rng.normal(0.20, 0.05, 50)
errors_b = rng.normal(0.17, 0.05, 50)

t_stat, p_value = stats.ttest_rel(errors_a, errors_b)
print(t_stat, p_value)
```

With these generating parameters (means 0.20 vs 0.17, both std 0.05, n=50),
the p-value should typically come out well below 0.05 — model B's lower error
is likely a real effect, not noise, and you'd reject `H0` ("no difference in
mean error").

## 4. Empirical Type I error rate

```python
rng = np.random.default_rng(2)
false_positives = 0
for _ in range(1000):
    a = rng.normal(0, 1, 30)
    b = rng.normal(0, 1, 30)   # same distribution as a -> H0 is true
    _, p = stats.ttest_ind(a, b)
    if p < 0.05:
        false_positives += 1

print(false_positives / 1000)   # should land close to 0.05
```

## 5. Chi-squared test

```python
table = np.array([
    [45, 55],
    [60, 40],
])
chi2, p, dof, expected = stats.chi2_contingency(table)
print(chi2, p)
```

With this table, the observed click rates (45% vs 60%) differ enough on
n=100-per-group that the p-value typically comes out below 0.05 — evidence
that click behavior is associated with variant, i.e. worth investigating B as
a real improvement over A rather than assuming the gap is noise.

## 6. Why "not significant" ≠ "no real difference"

Failing to reject `H0` only means the sample didn't provide *enough evidence*
to distinguish the observed effect from noise at your chosen `α` — it's
consistent both with "there truly is no effect" and with "there is a real but
small effect and/or your sample was too small to detect it" (low statistical
power). Absence of evidence is not evidence of absence; the correct next
question is usually "was this test well-powered to detect an effect of the
size I actually care about?" rather than concluding the effect doesn't exist.
