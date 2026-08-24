# 01 — Concepts: Derivatives & Chain Rule

## What a derivative is

The derivative `f'(x)` is the instantaneous rate of change of `f` at `x` —
the slope of the tangent line. Formally:

```
f'(x) = lim(h -> 0) [f(x+h) - f(x)] / h
```

In ML, if `f` is a loss function of a weight `w`, `f'(w)` tells you: "if I
nudge `w` up slightly, does the loss go up or down, and how fast?" — exactly
the signal gradient descent needs.

## Common derivative rules (the ones you'll actually use)

| Function | Derivative |
|---|---|
| `x^n` | `n * x^(n-1)` |
| `e^x` | `e^x` |
| `ln(x)` | `1/x` |
| `sin(x)` | `cos(x)` |
| constant | `0` |
| `c * f(x)` | `c * f'(x)` |
| `f(x) + g(x)` | `f'(x) + g'(x)` |
| `f(x) * g(x)` (product rule) | `f'(x)g(x) + f(x)g'(x)` |

## The chain rule — the single most important rule for ML

If `y = f(g(x))` (a function of a function), then:

```
dy/dx = f'(g(x)) * g'(x)
```

Read as: "the derivative of a composition is the product of the derivatives
of each piece." A neural network is a deep composition of functions (layer
after layer), so computing how the loss depends on an *early* layer's
weights requires chaining derivatives through every layer after it — this
chaining, applied systematically, **is** backpropagation (Lesson 037).

### Worked example

`y = sin(x^2)`. Let `u = x^2` (so `y = sin(u)`).

```
dy/du = cos(u) = cos(x^2)
du/dx = 2x
dy/dx = cos(x^2) * 2x
```

## Sigmoid derivative (you'll use this constantly)

`sigmoid(x) = 1 / (1 + e^-x)`. Its derivative has a famously clean form:

```
sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
```

This is why, once you've computed the sigmoid's output in a forward pass,
computing its derivative for backprop is nearly free — just reuse the output.

## Numerical differentiation (a sanity-check tool)

When you're not sure your hand-derived formula is correct, approximate the
derivative directly and compare:

```python
def numerical_derivative(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)
```

This **central difference** formula is what you'll use in Lesson 038 to
"gradient check" a from-scratch backprop implementation — if your analytical
gradient and the numerical one disagree by more than a tiny tolerance, your
backprop has a bug.

## Local minima, maxima, and critical points

Where `f'(x) = 0`, the function is momentarily flat — a candidate minimum,
maximum, or saddle point. Gradient descent (Lesson 015) works by repeatedly
moving *against* the gradient, converging toward points where the gradient
is (near) zero — hopefully a minimum of the loss, though for non-convex
functions (most neural network losses) it might be a saddle point or a local
rather than global minimum.
