# 03 — Solutions: Entropy, Cross-Entropy & KL Divergence

```python
import numpy as np

def entropy(p):
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    return -np.sum(p * np.log(p))
```

## 1. Fair vs biased coin

```python
print(entropy([0.5, 0.5]))   # ~0.693 nats (log 2)
print(entropy([0.9, 0.1]))   # ~0.325 nats
```

The fair coin has higher entropy — it's genuinely 50/50, maximally
unpredictable, while the biased coin's outcome (mostly heads) is easier to
guess correctly, hence lower average surprise.

## 2. Uniform maximizes entropy

```python
print(entropy([1/6]*6))                          # log(6) ~ 1.792
print(entropy([0.5, 0.1, 0.1, 0.1, 0.1, 0.1]))    # lower, e.g. ~1.34
print(np.log(6))                                   # confirms uniform = log(6), the maximum
```

## 3. Cross-entropy: correct vs confidently wrong

```python
def cross_entropy(p, q):
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    q = np.clip(q, 1e-12, 1)
    return -np.sum(p * np.log(q))

p = [1, 0, 0]
print(cross_entropy(p, [0.7, 0.2, 0.1]))   # -log(0.7) ~ 0.357 (low loss, correct+confident)
print(cross_entropy(p, [0.1, 0.2, 0.7]))   # -log(0.1) ~ 2.303 (high loss, confidently wrong)
```

## 4. KL divergence asymmetry

```python
def kl_divergence(p, q):
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    mask = p > 0
    return np.sum(p[mask] * np.log(p[mask] / q[mask]))

p, q = [0.5, 0.5], [0.9, 0.1]
print(kl_divergence(p, q))   # one value
print(kl_divergence(q, p))   # different value -> confirms asymmetry
```

## 5. Cross-entropy = entropy + KL identity

```python
rng = np.random.default_rng(0)
for _ in range(5):
    p_raw, q_raw = rng.uniform(0.1, 1, 3), rng.uniform(0.1, 1, 3)
    p, q = p_raw / p_raw.sum(), q_raw / q_raw.sum()
    lhs = cross_entropy(p, q)
    rhs = entropy(p) + kl_divergence(p, q)
    print(np.isclose(lhs, rhs))   # True every time
```

## 6. Loss vs confidence curve

```python
import matplotlib.pyplot as plt

probs = np.linspace(0.01, 0.99, 200)
losses = -np.log(probs)   # cross-entropy loss when true label is this class

plt.plot(probs, losses)
plt.xlabel("predicted probability of correct class")
plt.ylabel("cross-entropy loss")
plt.show()
```

As the predicted probability for the correct class approaches 0, `-log(q)`
diverges toward infinity — an unboundedly harsh penalty. Mean squared error
`(1-q)^2` on the same range only reaches a maximum of 1 — bounded, much
gentler. This is a deliberate design choice: cross-entropy pushes the model
much harder to avoid confident wrong predictions, which is exactly the
failure mode you want penalized heavily in a classifier or language model.
