# 03 — Solutions: Derivatives & Chain Rule

## 1. Polynomial

`f'(x) = 6x + 2`

## 2. Exponential via chain rule

`u = 2x`, `f = e^u`. `df/du = e^u`, `du/dx = 2`. So `f'(x) = 2 * e^(2x)`.

## 3. Chain rule with a power

`u = x^2 + 1`, `f = u^3`. `df/du = 3u^2`, `du/dx = 2x`.
So `f'(x) = 3(x^2+1)^2 * 2x = 6x(x^2+1)^2`.

## 4. Sigmoid derivative derivation

```
sigmoid(x) = (1 + e^-x)^-1

Let u = 1 + e^-x, so sigmoid = u^-1.
d(sigmoid)/du = -u^-2
du/dx = -e^-x

d(sigmoid)/dx = -u^-2 * (-e^-x) = e^-x / (1 + e^-x)^2
```

Now rewrite in terms of `sigmoid(x)` itself:

```
sigmoid(x) * (1 - sigmoid(x))
  = [1/(1+e^-x)] * [1 - 1/(1+e^-x)]
  = [1/(1+e^-x)] * [e^-x/(1+e^-x)]
  = e^-x / (1+e^-x)^2
```

Matches — confirming `sigmoid'(x) = sigmoid(x)(1 - sigmoid(x))`.

## 5. Numerical check

```python
def numerical_derivative(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)

f1 = lambda x: 3*x**2 + 2*x - 5
f2 = lambda x: np.exp(2*x)
f3 = lambda x: (x**2 + 1)**3

print(numerical_derivative(f1, 2), "vs analytical", 6*2 + 2)              # ~14
print(numerical_derivative(f2, 2), "vs analytical", 2*np.exp(4))         # match
print(numerical_derivative(f3, 2), "vs analytical", 6*2*(2**2+1)**2)     # match
```

## 6. Sigmoid derivative check

```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    s = sigmoid(x)
    return s * (1 - s)

for x in [-2, 0, 1, 5]:
    print(sigmoid_prime(x), numerical_derivative(sigmoid, x))
```

Both should agree to several decimal places for every value of `x`.

## 7. Gradient-check utility

```python
def gradient_check(f, f_prime, n=5, seed=0, tol=1e-4):
    rng = np.random.default_rng(seed)
    for x in rng.uniform(-5, 5, n):
        analytical = f_prime(x)
        numerical = numerical_derivative(f, x)
        assert abs(analytical - numerical) < tol, (x, analytical, numerical)
    print("all checks passed")

gradient_check(lambda x: x**3, lambda x: 3*x**2)
```

This exact pattern — compare an analytical gradient to a numerical
approximation on random points — is the standard way to catch bugs in a
from-scratch backpropagation implementation (Lesson 038) before trusting it
on real data.
