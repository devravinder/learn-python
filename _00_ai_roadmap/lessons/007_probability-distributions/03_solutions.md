# 03 — Solutions: Probability Distributions

## 1. Binomial approaching Normal

```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
samples = rng.binomial(n=20, p=0.5, size=10000)

plt.hist(samples, bins=range(0, 22))
plt.show()
```

As `n` grows, the Binomial distribution's shape approaches a bell curve
(Normal distribution) — a special case of the Central Limit Theorem, since a
Binomial is itself a sum of `n` independent Bernoulli trials.

## 2. Empirical rule check

```python
rng = np.random.default_rng(0)
samples = rng.normal(0, 1, 10000)

for k in [1, 2, 3]:
    frac = (np.abs(samples) < k).mean()
    print(f"within {k} std: {frac:.4f}")
# expect roughly 0.68, 0.95, 0.997
```

## 3. Z-scores

```python
import numpy as np

scores = np.array([55, 60, 62, 70, 72, 75, 80, 85, 90, 95])
mu, sigma = scores.mean(), scores.std()
z = (scores - mu) / sigma
print(dict(zip(scores, z.round(2))))
```

The score with the largest absolute z-score is the most unusual relative to
the group (typically the min or max here — with this data, 95 has the largest
positive z-score and 55 the largest negative).

## 4. Softmax from scratch

```python
def softmax(z):
    exp_z = np.exp(z)
    return exp_z / exp_z.sum()

z = np.array([2.0, 1.0, 0.1])
probs = softmax(z)
print(probs, probs.sum())   # sums to 1.0
```

## 5. Numerically stable softmax

```python
z = np.array([1000., 1001., 1002.])
softmax(z)   # exp(1000) overflows -> inf/inf -> nan
```

`exp(1000)` overflows float64's range, producing `inf`, and `inf/inf` is
`nan`. Fix:

```python
def softmax(z):
    shifted = z - np.max(z)
    exp_z = np.exp(shifted)
    return exp_z / exp_z.sum()
```

Subtracting a constant `c = max(z)` from every element doesn't change the
result mathematically, because `exp(z_i - c) / Σ exp(z_j - c) = exp(z_i) /
Σ exp(z_j)` (the `exp(-c)` factor cancels top and bottom) — but it keeps the
largest exponent at `exp(0) = 1`, avoiding overflow. This exact trick is used
inside every real softmax implementation, including PyTorch's.

## 6. Sum of two dice

```python
rng = np.random.default_rng(0)
rolls = rng.integers(1, 7, size=(100_000, 2))
sums = rolls.sum(axis=1)

plt.hist(sums, bins=range(2, 14))
plt.show()
```

The result is a triangular shape (peaking at 7, since there are more ways to
roll a sum of 7 than any other value), already visibly closer to bell-shaped
than a single die's flat/uniform histogram — with more dice summed, it
converges further toward a true Normal curve.
