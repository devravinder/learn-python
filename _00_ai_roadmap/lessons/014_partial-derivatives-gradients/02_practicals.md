# 02 — Practicals: Partial Derivatives, Gradients & Jacobians

## Pen-and-paper

1. For `f(x, y) = 3x^2 + 2xy + y^2`, compute `∂f/∂x` and `∂f/∂y`, then write
   `∇f(x, y)`.

2. Evaluate `∇f` from Q1 at the point `(1, 2)`. In which direction does `f`
   increase fastest from that point?

3. For `f(x, y) = x^2 + y^2` (a paraboloid bowl), compute `∇f` and confirm
   that at `(0, 0)` the gradient is the zero vector — consistent with
   `(0,0)` being the function's minimum.

## Verify numerically in code

4. Implement a numerical gradient function:
   ```python
   def numerical_gradient(f, point, h=1e-5):
       grad = np.zeros_like(point, dtype=float)
       for i in range(len(point)):
           point_plus = point.copy(); point_plus[i] += h
           point_minus = point.copy(); point_minus[i] -= h
           grad[i] = (f(*point_plus) - f(*point_minus)) / (2 * h)
       return grad
   ```
   Use it to check your hand-computed gradients from Q1 and Q2 at `(1, 2)`.

5. For `f(x, y) = x^2 * y + y^3` from the concepts doc, implement both the
   analytical gradient and the numerical gradient, and confirm they match at
   3 different random points.

6. **Gradient ascent by hand (3 steps)**: starting at `(1.0, 1.0)`, for
   `f(x, y) = -(x^2 + y^2)` (a downward bowl, maximized at the origin), take
   3 steps of gradient *ascent* (`point += 0.1 * grad_f(point)`) using your
   numerical gradient function. Print the point after each step — it should
   move toward `(0, 0)`.
