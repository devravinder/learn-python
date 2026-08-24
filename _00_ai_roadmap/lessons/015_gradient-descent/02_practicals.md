# 02 — Practicals: Gradient Descent

Use this synthetic linear dataset:

```python
import numpy as np
rng = np.random.default_rng(0)
x = rng.uniform(0, 10, 100)
y = 3 * x + 7 + rng.normal(0, 1, 100)   # true relationship: y = 3x + 7
```

1. Implement batch gradient descent (from `01_concepts.md`) for
   `ŷ = w*x + b` on this data. Train for 1000 epochs at `lr=0.01`. Print the
   final `w`, `b` — are they close to the true 3 and 7?

2. Track the loss (MSE) every 100 epochs during training and plot the loss
   curve. Does it decrease monotonically?

3. Repeat training with `lr = 0.001`, `lr = 0.01`, and `lr = 0.1` (500 epochs
   each). Plot all three loss curves on one chart. Which converges fastest?
   Does the largest learning rate diverge or oscillate?

4. Implement **stochastic gradient descent**: at each step, pick one random
   `(x_i, y_i)` pair and update `w, b` from just that single example's
   gradient. Run for 2000 steps and compare the final `w, b` and the loss
   curve's smoothness to batch gradient descent's.

5. Implement **mini-batch gradient descent** with batch size 16. Compare
   convergence speed and final loss to both batch and stochastic versions.

6. Rescale `x` to have mean 0 and std 1 before training (keep `y`
   unscaled). Retrain with batch gradient descent at the *same* learning
   rate you used in Q1. Does it converge faster, slower, or diverge? Explain
   why, referencing feature scaling from `01_concepts.md`. (Note: you'll need
   to adjust how you interpret the learned `w` back to the original scale,
   or just compare convergence speed rather than final coefficients.)
