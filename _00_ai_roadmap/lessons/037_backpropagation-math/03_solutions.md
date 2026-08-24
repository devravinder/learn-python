# 03 — Solutions: Backpropagation

## 1. Redo with x=1

```
Forward: z1 = 0.5*1 = 0.5, a1 = relu(0.5) = 0.5, z2 = 1.0*0.5 = 0.5, ŷ = 0.5
L = (0.5 - 3)^2 = 6.25

Backward:
∂L/∂ŷ = 2*(0.5-3) = -5.0
∂L/∂z2 = -5.0
∂L/∂w2 = -5.0 * a1 = -5.0 * 0.5 = -2.5
∂L/∂b2 = -5.0
∂L/∂a1 = -5.0 * w2 = -5.0 * 1.0 = -5.0
∂L/∂z1 = -5.0 * relu'(0.5) = -5.0 * 1 = -5.0
∂L/∂w1 = -5.0 * x = -5.0 * 1 = -5.0
∂L/∂b1 = -5.0
```

## 2. Computation graph for (a+b)(a-b)

```
Let u = a+b, v = a-b, f = u*v

∂f/∂u = v = a-b
∂f/∂v = u = a+b

∂u/∂a = 1, ∂u/∂b = 1
∂v/∂a = 1, ∂v/∂b = -1

∂f/∂a = ∂f/∂u * ∂u/∂a + ∂f/∂v * ∂v/∂a = (a-b)*1 + (a+b)*1 = 2a
∂f/∂b = ∂f/∂u * ∂u/∂b + ∂f/∂v * ∂v/∂b = (a-b)*1 + (a+b)*(-1) = -2b
```

Direct check: `f = a² - b²`, so `∂f/∂a = 2a`, `∂f/∂b = -2b` — matches exactly.

## 3. NumPy forward/backward

```python
import numpy as np

def relu(x): return np.maximum(0, x)
def relu_prime(x): return (x > 0).astype(float)

def forward(x, w1, b1, w2, b2):
    z1 = w1 * x + b1
    a1 = relu(z1)
    z2 = w2 * a1 + b2
    return z2, (x, z1, a1)

def backward(y_hat, y, w2, cache):
    x, z1, a1 = cache
    dL_dyhat = 2 * (y_hat - y)
    dL_dz2 = dL_dyhat * 1
    dL_dw2 = dL_dz2 * a1
    dL_db2 = dL_dz2
    dL_da1 = dL_dz2 * w2
    dL_dz1 = dL_da1 * relu_prime(z1)
    dL_dw1 = dL_dz1 * x
    dL_db1 = dL_dz1
    return dL_dw1, dL_db1, dL_dw2, dL_db2

x, w1, b1, w2, b2, y = 2, 0.5, 0, 1.0, 0, 3
y_hat, cache = forward(x, w1, b1, w2, b2)
grads = backward(y_hat, y, w2, cache)
print(grads)   # (-8.0, -4.0, -4.0, -4.0), matching 01_concepts.md exactly
```

## 4. Numerical gradient check

```python
def loss_fn(w1, b1, w2, b2):
    y_hat, _ = forward(x, w1, b1, w2, b2)
    return (y_hat - y) ** 2

def numerical_grad(param_name, params, h=1e-5):
    p = dict(params)
    p[param_name] += h
    loss_plus = loss_fn(**p)
    p[param_name] -= 2 * h
    loss_minus = loss_fn(**p)
    return (loss_plus - loss_minus) / (2 * h)

params = {"w1": w1, "b1": b1, "w2": w2, "b2": b2}
analytical = dict(zip(["w1", "b1", "w2", "b2"], grads))
for name in params:
    num = numerical_grad(name, params)
    print(name, "analytical:", analytical[name], "numerical:", num)
```

Both should match to about 4-5 decimal places, confirming the from-scratch
backward pass is correctly implemented.

## 5. Batched version

```python
def forward_batch(X, w1, b1, w2, b2):
    z1 = w1 * X + b1
    a1 = relu(z1)
    z2 = w2 * a1 + b2
    return z2, (X, z1, a1)

def backward_batch(y_hat, y, w2, cache):
    X, z1, a1 = cache
    n = len(X)
    dL_dyhat = 2 * (y_hat - y) / n     # averaged over batch
    dL_dz2 = dL_dyhat
    dL_dw2 = np.sum(dL_dz2 * a1)
    dL_db2 = np.sum(dL_dz2)
    dL_da1 = dL_dz2 * w2
    dL_dz1 = dL_da1 * relu_prime(z1)
    dL_dw1 = np.sum(dL_dz1 * X)
    dL_db1 = np.sum(dL_dz1)
    return dL_dw1, dL_db1, dL_dw2, dL_db2

X_batch = np.array([2.0, 1.0, 3.0])
y_batch = np.array([3.0, 3.0, 3.0])
y_hat_batch, cache_batch = forward_batch(X_batch, w1, b1, w2, b2)
batch_grads = backward_batch(y_hat_batch, y_batch, w2, cache_batch)

# manual average of 3 single-sample gradients
single_grads = []
for xi, yi in zip(X_batch, y_batch):
    yh, c = forward(xi, w1, b1, w2, b2)
    single_grads.append(backward(yh, yi, w2, c))
manual_avg = np.mean(single_grads, axis=0)

print(batch_grads)
print(manual_avg)   # should match closely
```

## 6. Branching gradients

```python
def L_and_grad(x):
    y1 = x**2
    y2 = 3*x
    L = y1 + y2
    dL_dy1 = 1
    dL_dy2 = 1
    dy1_dx = 2*x
    dy2_dx = 3
    dL_dx = dL_dy1 * dy1_dx + dL_dy2 * dy2_dx
    return L, dL_dx

x = 4
L, dL_dx = L_and_grad(x)
direct_derivative = 2*x + 3
print(dL_dx, direct_derivative)   # both 11 -> match
```

Summing gradients across both paths (`x**2`'s path and `3*x`'s path) gives
the exact same result as directly differentiating the combined expression
`x**2 + 3*x` — confirming the "sum gradients at a branch" rule from
`01_concepts.md` is correct, not just a convenient shortcut.
