# 03 — Solutions: Partial Derivatives, Gradients & Jacobians

## 1. Partial derivatives and gradient

```
f(x, y) = 3x^2 + 2xy + y^2

∂f/∂x = 6x + 2y
∂f/∂y = 2x + 2y

∇f(x, y) = [6x + 2y, 2x + 2y]
```

## 2. Gradient at (1, 2)

```
∇f(1, 2) = [6(1) + 2(2), 2(1) + 2(2)] = [10, 6]
```

`f` increases fastest in the direction of the vector `[10, 6]` (or its
normalized unit-vector version) from that point — that's the definition of
the gradient as the steepest-ascent direction.

## 3. Gradient at the minimum

```
f(x, y) = x^2 + y^2
∇f = [2x, 2y]
∇f(0, 0) = [0, 0]
```

A zero gradient at the minimum matches the single-variable idea that
`f'(x) = 0` at critical points — here, in every direction simultaneously,
confirming `(0,0)` is (at least) a critical point, and since the bowl curves
upward everywhere, it's a true minimum.

## 4–5. Numerical gradient checks

```python
import numpy as np

def numerical_gradient(f, point, h=1e-5):
    point = np.array(point, dtype=float)
    grad = np.zeros_like(point)
    for i in range(len(point)):
        point_plus = point.copy(); point_plus[i] += h
        point_minus = point.copy(); point_minus[i] -= h
        grad[i] = (f(*point_plus) - f(*point_minus)) / (2 * h)
    return grad

f1 = lambda x, y: 3*x**2 + 2*x*y + y**2
print(numerical_gradient(f1, [1, 2]))   # ~[10, 6]

f2 = lambda x, y: x**2 * y + y**3
def analytical_grad_f2(x, y):
    return np.array([2*x*y, x**2 + 3*y**2])

rng = np.random.default_rng(0)
for _ in range(3):
    pt = rng.uniform(-5, 5, 2)
    a = analytical_grad_f2(*pt)
    n = numerical_gradient(f2, pt)
    print(np.allclose(a, n, atol=1e-3))   # True
```

## 6. Gradient ascent toward the maximum

```python
def f(x, y):
    return -(x**2 + y**2)

point = np.array([1.0, 1.0])
for step in range(3):
    grad = numerical_gradient(f, point)
    point = point + 0.1 * grad
    print(step, point)
```

Each step moves `point` closer to `[0, 0]`, since the gradient of
`-(x^2+y^2)` points back toward the origin (the function's maximum) from
anywhere — this 3-line loop *is* the entire mechanical idea behind gradient
descent/ascent; Lesson 015 formalizes it and applies it to a real loss
function instead of a toy bowl.
