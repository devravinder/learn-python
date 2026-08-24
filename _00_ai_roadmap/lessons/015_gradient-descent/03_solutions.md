# 03 — Solutions: Gradient Descent

```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
x = rng.uniform(0, 10, 100)
y = 3 * x + 7 + rng.normal(0, 1, 100)


def mse(x, y, w, b):
    y_hat = w * x + b
    return np.mean((y_hat - y) ** 2)
```

## 1. Batch gradient descent

```python
def train_batch(x, y, lr=0.01, epochs=1000):
    w, b = 0.0, 0.0
    n = len(x)
    for _ in range(epochs):
        y_hat = w * x + b
        error = y_hat - y
        dw = (2 / n) * np.dot(error, x)
        db = (2 / n) * error.sum()
        w -= lr * dw
        b -= lr * db
    return w, b

w, b = train_batch(x, y, lr=0.01, epochs=1000)
print(w, b)   # should land close to 3.0 and 7.0
```

## 2. Loss curve

```python
def train_batch_tracked(x, y, lr=0.01, epochs=1000):
    w, b = 0.0, 0.0
    n = len(x)
    losses = []
    for epoch in range(epochs):
        y_hat = w * x + b
        error = y_hat - y
        dw = (2 / n) * np.dot(error, x)
        db = (2 / n) * error.sum()
        w -= lr * dw
        b -= lr * db
        if epoch % 100 == 0:
            losses.append(mse(x, y, w, b))
    return w, b, losses

w, b, losses = train_batch_tracked(x, y)
plt.plot(losses)
plt.show()
```

Loss should decrease monotonically for batch gradient descent on this convex
problem at a reasonable learning rate — each step is guaranteed not to
increase the loss (for small-enough `lr`) since it moves exactly along the
true gradient.

## 3. Learning rate comparison

```python
fig, ax = plt.subplots()
for lr in [0.001, 0.01, 0.1]:
    _, _, losses = train_batch_tracked(x, y, lr=lr, epochs=500)
    ax.plot(losses, label=f"lr={lr}")
ax.legend()
ax.set_yscale("log")
plt.show()
```

`lr=0.01` or `lr=0.1` should converge fastest (fewer epochs to low loss);
depending on the exact data scale, `lr=0.1` may start oscillating visibly in
the loss curve, and a too-large `lr` (try `lr=0.5`) will typically diverge
outright (loss shoots to very large or `nan` values) — a good demonstration
to add if you want to see divergence explicitly.

## 4. Stochastic gradient descent

```python
def train_sgd(x, y, lr=0.01, steps=2000, seed=0):
    rng = np.random.default_rng(seed)
    w, b = 0.0, 0.0
    for _ in range(steps):
        i = rng.integers(0, len(x))
        xi, yi = x[i], y[i]
        error = w * xi + b - yi
        w -= lr * 2 * error * xi
        b -= lr * 2 * error
    return w, b

w_sgd, b_sgd = train_sgd(x, y)
print(w_sgd, b_sgd)   # close to 3, 7, but noisier per-step than batch
```

SGD's loss curve (if tracked) is visibly jagged step-to-step compared to
batch gradient descent's smooth monotonic decrease, because each step only
sees one noisy sample's gradient rather than the true average gradient.

## 5. Mini-batch gradient descent

```python
def train_minibatch(x, y, lr=0.01, epochs=200, batch_size=16, seed=0):
    rng = np.random.default_rng(seed)
    w, b = 0.0, 0.0
    n = len(x)
    for _ in range(epochs):
        idx = rng.permutation(n)
        for start in range(0, n, batch_size):
            batch_idx = idx[start:start + batch_size]
            xb, yb = x[batch_idx], y[batch_idx]
            error = w * xb + b - yb
            dw = (2 / len(xb)) * np.dot(error, xb)
            db = (2 / len(xb)) * error.sum()
            w -= lr * dw
            b -= lr * db
    return w, b

w_mb, b_mb = train_minibatch(x, y)
print(w_mb, b_mb)
```

Mini-batch typically lands between batch and SGD: smoother than SGD (each
step averages over 16 samples, reducing noise) but faster per-epoch-wall-clock
than full batch on large datasets — the standard tradeoff that makes it the
default choice for deep learning.

## 6. Effect of feature scaling

```python
x_scaled = (x - x.mean()) / x.std()
w2, b2, losses2 = train_batch_tracked(x_scaled, y, lr=0.01, epochs=1000)

plt.plot(losses, label="unscaled x")
plt.plot(losses2, label="scaled x")
plt.legend()
plt.yscale("log")
plt.show()
```

With `x` unscaled (range roughly 0–10), the loss surface is stretched, so a
learning rate that's stable often converges slowly; with `x` standardized,
the same `lr` typically converges dramatically faster to a low loss, because
the loss surface becomes better-conditioned (closer to a circular bowl
instead of a stretched valley) — a direct, hands-on confirmation of why
feature scaling before training is standard practice.
