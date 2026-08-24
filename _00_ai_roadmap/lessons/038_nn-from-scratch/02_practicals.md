# 02 — Practicals: Neural Network From Scratch

Build this incrementally — each part depends on the last, exactly like
building up micrograd step by step.

1. Implement the `Value` class from `01_concepts.md` with `__add__` and
   `__mul__`. Also implement `__pow__` (for `x**n`, needed for MSE loss and
   `1/x` via `x**-1`), `relu()`, and `tanh()` (derivative:
   `1 - tanh(x)^2`, Lesson 013). Add the convenience dunders
   (`__neg__`, `__sub__`, `__radd__`, `__rmul__`, `__truediv__`) so
   expressions like `a - b` and `2 * a` work naturally.

2. Implement `backward()` using the topological-sort approach from
   `01_concepts.md`. Test it on the exact tiny network from Lesson 037's
   worked example (`x=2, w1=0.5, b1=0, w2=1.0, b2=0, y=3`) and confirm your
   computed gradients (`w1.grad, b1.grad, w2.grad, b2.grad`) exactly match
   Lesson 037's hand-derived values (`-8.0, -4.0, -4.0, -4.0`).

3. Implement `Neuron`, `Layer`, and `MLP` from `01_concepts.md`. Give
   `Neuron`/`Layer`/`MLP` a `nonlin` option so you can build hidden layers
   with `tanh` and a **linear** (no activation) output layer — regression-
   style outputs shouldn't be squashed.

4. Train an `MLP(2, [4, 4, 1])` on XOR, using targets `-1.0`/`1.0` (not
   `0`/`1` — friendlier for a `tanh`-based network) and MSE loss. Zero every
   parameter's `.grad` before each `backward()` call (they accumulate via
   `+=`, so stale gradients from the previous step must be cleared). Train
   for a few hundred epochs and confirm the loss approaches 0 and final
   predictions match the targets closely.

5. Plot the loss curve over training. Does it decrease smoothly, similar to
   Lesson 015's gradient descent curves?

6. **Gradient-check your engine**: pick any single weight in the trained
   model, perturb it by `h=1e-4`, recompute the total loss, and compare the
   numerical gradient to `weight.grad` from your last `backward()` call
   (Lesson 013's central-difference check). Confirm they match closely —
   this is the same sanity check real deep learning framework authors run
   when they write a new autograd engine.
