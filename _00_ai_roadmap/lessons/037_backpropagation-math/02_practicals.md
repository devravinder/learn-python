# 02 — Practicals: Backpropagation

## Pen-and-paper (extend the worked example)

1. Using the same tiny network from `01_concepts.md`
   (`x=2, w1=0.5, b1=0, w2=1.0, b2=0, y=3`), redo the full forward and
   backward pass but with `x=1` instead. Compute all the same gradients
   (`∂L/∂w1, ∂L/∂b1, ∂L/∂w2, ∂L/∂b2`).

2. Draw the computation graph for `f(a, b) = (a + b) * (a - b)` (i.e.
   `a² - b²`, though pretend you don't know that identity). Compute
   `∂f/∂a` and `∂f/∂b` by hand via the graph, then verify against the direct
   derivative of `a² - b²`.

## Code

3. Implement the tiny 2-layer network from `01_concepts.md` in NumPy: a
   `forward(x)` function that also **caches** every intermediate value
   (`z1, a1, z2`), and a `backward()` function that uses those cached values
   to compute all 4 gradients. Confirm your code reproduces the exact
   numbers from the worked example.

4. Verify your `backward()` function using the numerical gradient check
   from Lesson 013/014: perturb each weight by a tiny `h`, recompute the
   loss, estimate the gradient numerically, and confirm it matches your
   analytical gradient within `1e-4`.

5. Extend Q3's network to a batch of 3 samples at once (vectorized, per the
   matrix-form equations at the end of `01_concepts.md`). Confirm your
   batched implementation, averaged, produces gradients consistent with
   running the single-sample version 3 times and averaging by hand.

6. Implement the branching case from `01_concepts.md`: a value `x` used in
   two places, `y1 = x**2` and `y2 = 3*x`, combined as `L = y1 + y2`.
   Compute `∂L/∂x` by (a) summing the two paths' gradients per the
   multivariable chain rule, and (b) directly differentiating
   `L = x**2 + 3*x`. Confirm they match.
