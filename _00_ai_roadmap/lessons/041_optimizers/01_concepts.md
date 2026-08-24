# 01 — Concepts: Optimizers

## Why plain SGD isn't enough

Lesson 015's plain gradient descent uses the same learning rate for every
parameter, every step. Real loss landscapes are uneven — steep in some
directions, flat in others — so a single learning rate is either too slow
in flat regions or too unstable in steep ones. Every optimizer below fixes
this in a different way.

## Momentum: smoothing the path with velocity

```
v = β*v + (1-β)*gradient        # exponentially-weighted moving average of gradients
w = w - lr*v
```

Instead of stepping directly along the current gradient, accumulate a
"velocity" — like a ball rolling downhill that keeps some of its previous
direction. This smooths out noisy gradients (helpful for SGD's per-sample
noise, Lesson 015) and speeds through consistent-direction regions while
damping oscillation in narrow valleys (exactly the "zig-zag" problem
unscaled/correlated features caused in Lesson 015).

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```

## RMSProp: per-parameter adaptive learning rates

```
s = β*s + (1-β)*gradient^2                  # moving average of squared gradients
w = w - lr * gradient / (sqrt(s) + ε)
```

Divides each parameter's update by the root-mean-square of its *own* recent
gradient magnitude — parameters with consistently large gradients get
smaller effective steps (preventing overshoot), parameters with small
gradients get relatively larger steps (preventing them from stalling). This
directly addresses the "one learning rate for wildly different-scaled
gradients" problem.

## Adam: momentum + RMSProp combined (the modern default)

Adam tracks both a momentum-style moving average of gradients (`m`, "first
moment") and an RMSProp-style moving average of squared gradients (`v`,
"second moment"), with a bias correction for early training steps (both
start at zero, biasing early estimates toward zero without the correction):

```
m = β1*m + (1-β1)*gradient
v = β2*v + (1-β2)*gradient^2
m_hat = m / (1 - β1^t)     # bias correction
v_hat = v / (1 - β2^t)
w = w - lr * m_hat / (sqrt(v_hat) + ε)
```

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)   # (β1=0.9, β2=0.999 by default)
```

**Adam is the default choice for most deep learning today**, including LLM
pretraining (often a variant called **AdamW**, which decouples weight decay
from the adaptive update in a way that works better in practice — the
standard optimizer for Transformers, Lesson 060+).

## Learning rate schedules

Even with an adaptive optimizer, the learning rate itself is often varied
over training:

- **Warmup**: start with a very small learning rate and ramp up over the
  first few hundred/thousand steps — stabilizes early training before
  gradient statistics (Adam's moving averages) have accumulated enough
  signal to be reliable. Standard practice for Transformer training.
- **Decay** (cosine, step, or linear): reduce the learning rate over the
  course of training — large steps early for fast progress, small steps
  late for fine convergence, echoing Lesson 015's "too-large learning rate
  overshoots near the minimum" concern.

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
for epoch in range(num_epochs):
    train_one_epoch(...)
    scheduler.step()
```

## Practical guidance

- **Adam/AdamW** is a safe, strong default almost everywhere in deep
  learning, especially for Transformers/LLMs.
- **SGD with momentum** sometimes generalizes slightly better for
  vision/CNN tasks (Lesson 043) with careful learning rate tuning, and
  remains common there.
- Typical Adam learning rates: `1e-3` to `1e-4` for smaller networks,
  `1e-4` to `5e-4` for Transformer pretraining (with warmup).
- If loss diverges (goes to `nan`/`inf`): learning rate is very likely too
  high — the same diagnosis as Lesson 015, regardless of which optimizer
  you're using.
