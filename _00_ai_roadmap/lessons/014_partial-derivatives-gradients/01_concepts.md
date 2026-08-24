# 01 — Concepts: Partial Derivatives, Gradients & Jacobians

## Partial derivatives

For a function of multiple variables, `f(x, y)`, the **partial derivative**
`∂f/∂x` treats every other variable as a constant and differentiates only
with respect to `x`.

Example: `f(x, y) = x^2 * y + y^3`

```
∂f/∂x = 2xy         (treat y as constant)
∂f/∂y = x^2 + 3y^2   (treat x as constant)
```

## The gradient

The **gradient** `∇f` is the vector of all partial derivatives:

```
∇f(x, y) = [∂f/∂x, ∂f/∂y]
```

Geometrically, the gradient points in the direction of **steepest ascent** —
the direction that increases `f` fastest from the current point. This is
exactly why gradient *descent* moves in the **negative** gradient direction:
that's steepest decrease.

For a loss function `L(w1, w2, ..., wn)` over `n` weights, `∇L` is an
`n`-dimensional vector — one entry per weight, telling you exactly how to
nudge each weight to reduce the loss fastest.

```python
import numpy as np

def f(x, y):
    return x**2 * y + y**3

def grad_f(x, y):
    return np.array([2*x*y, x**2 + 3*y**2])
```

## The Jacobian

When a function maps a vector to another **vector** (not just a scalar), the
**Jacobian** `J` is the matrix of all partial derivatives — row `i`, column
`j` is `∂(output_i)/∂(input_j)`:

```
f: R^n -> R^m
J[i, j] = ∂f_i / ∂x_j     (shape: m x n)
```

The gradient is a special case: when `m=1` (scalar output), the Jacobian is
just a single row — the gradient (transposed). Every neural network layer is
a vector-to-vector function, so backpropagation through a network is really
Jacobian-vector products chained together via the multivariable chain rule.

## Multivariable chain rule

If `z = f(x, y)` and both `x = x(t)` and `y = y(t)` depend on some `t`:

```
dz/dt = (∂f/∂x)(dx/dt) + (∂f/∂y)(dy/dt)
```

This generalizes the single-variable chain rule by summing contributions
through every path the dependency flows — exactly the rule backpropagation
applies at every node of a computation graph (Lesson 037), since a weight
deep in a network can affect the loss through multiple downstream paths.

## Directional derivative

The rate of change of `f` in an arbitrary direction `u` (unit vector) is
`∇f · u` — the dot product of the gradient with that direction. This
confirms the gradient's "steepest ascent" property: `∇f · u` is maximized
when `u` points in the same direction as `∇f` itself (since
`a·b = ||a||||b||cos θ` is maximized at `θ=0`).

## Second derivatives: the Hessian (preview)

The **Hessian** is the matrix of all second partial derivatives — it
describes the *curvature* of a function. A positive-definite Hessian
(all eigenvalues positive, Lesson 012) means the function curves upward in
every direction — a true local minimum. Some advanced optimizers (Newton's
method, and approximations used in some LLM training setups) use Hessian
information; plain gradient descent (Lesson 015) doesn't need it, but
knowing it exists explains why loss landscapes with different curvatures in
different directions are harder to optimize.
