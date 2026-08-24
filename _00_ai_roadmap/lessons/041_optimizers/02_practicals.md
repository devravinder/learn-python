# 02 — Practicals: Optimizers

Use a deliberately ill-conditioned toy loss surface to make optimizer
differences visible:

```python
import torch

def rosenbrock(xy):
    x, y = xy[0], xy[1]
    return (1 - x)**2 + 100 * (y - x**2)**2   # classic optimizer-torture-test function
```

1. Starting from `xy = torch.tensor([-1.5, 2.0], requires_grad=True)`, run
   200 steps of plain `torch.optim.SGD(lr=0.001)` minimizing `rosenbrock`.
   Track the loss and the `(x, y)` path. Does it make meaningful progress
   toward the true minimum at `(1, 1)`?

2. Repeat with `SGD(lr=0.001, momentum=0.9)`. Compare the final loss and
   path to Q1 — does momentum help escape the narrow curved valley faster?

3. Repeat with `torch.optim.Adam(lr=0.05)`. Compare convergence speed to
   Q1 and Q2.

4. Plot all three optimizers' loss curves (log scale) on one chart. Which
   reaches the lowest loss in 200 steps?

5. On the XOR MLP from Lesson 040, compare final loss after a fixed 100
   epochs across `SGD(lr=0.1)`, `SGD(lr=0.1, momentum=0.9)`, `RMSprop(lr=0.01)`,
   and `Adam(lr=0.01)`. Report a small table of results.

6. Add a cosine annealing learning rate schedule
   (`torch.optim.lr_scheduler.CosineAnnealingLR`) to the Adam run from Q5,
   with `T_max=100`. Plot the learning rate over training
   (`scheduler.get_last_lr()` each epoch) to confirm it actually decays as
   expected, and compare final loss to the non-scheduled Adam run.
