# 01 — Concepts: Dropout & BatchNorm

## Dropout: forcing redundancy

During training, **randomly zero out a fraction `p` of a layer's
activations** on each forward pass (different random units each time):

```python
import torch.nn as nn
layer = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Dropout(p=0.5))
```

Why this helps: a unit can't rely on any *specific* other unit always being
present (it might get dropped next batch), forcing the network to learn
more redundant, robust representations instead of fragile co-adapted
features — conceptually similar to Random Forest's feature subsetting
(Lesson 027) decorrelating trees, but applied to a single network's units
across different forward passes instead of across separate trees.

**Critical detail — train vs eval behavior**: dropout is active during
training but **must be disabled** at inference (`model.eval()`, Lesson 040) —
at eval time, all units are used, but their outputs are scaled by `(1-p)`
during training (**inverted dropout**, what PyTorch does automatically) so
the expected magnitude of activations matches between train and eval without
needing any rescaling at inference time.

```python
model.train()   # dropout active
model.eval()    # dropout disabled, scaling handled automatically
```

## Batch Normalization: stabilizing layer inputs

Normalizes each layer's activations (per mini-batch) to have mean 0,
variance 1, then applies a learnable scale (`γ`) and shift (`β`) so the
network can undo the normalization if that's actually better for a specific
layer:

```
BN(x) = γ * (x - batch_mean) / sqrt(batch_var + ε) + β
```

```python
layer = nn.Sequential(nn.Linear(64, 64), nn.BatchNorm1d(64), nn.ReLU())
```

Why this helps: keeps activation distributions stable across layers and
training steps (reduces "internal covariate shift" — later layers don't have
to constantly readjust to shifting input distributions from earlier layers
as they update), which in practice enables **higher learning rates** and
**faster convergence**, and provides a mild regularization effect (the
batch-based statistics add a bit of noise, similar in spirit to dropout).

**Train vs eval behavior, again matters**: at training time, BatchNorm uses
the *current batch's* mean/variance; at eval time, it uses a running average
of mean/variance accumulated during training (since a single test example,
or a differently-sized batch, shouldn't have its normalization depend on
whatever else happens to be in that particular batch). `model.eval()`
switches this automatically — another reason that call is essential, not
just for dropout.

## Where to place BatchNorm relative to the activation

Common convention: `Linear -> BatchNorm -> Activation` (normalize the raw
linear output, then apply the nonlinearity) — though `Linear -> Activation
-> BatchNorm` also appears in practice; both are used, and the "correct"
order is somewhat debated/task-dependent. What matters more is being
consistent and validating empirically for your specific problem.

## Dropout and BatchNorm together — often redundant

Since BatchNorm already has a mild regularizing effect and the two can
interact in ways that sometimes hurt rather than help (there's research
showing they can conflict on their moving-average statistics),
many modern architectures use one or the other, not both indiscriminately
— and many modern Transformer architectures (Lesson 060) use **LayerNorm**
(normalizes across features for a single example, not across the batch)
rather than BatchNorm at all, since BatchNorm's batch-dependent statistics
don't fit variable-length sequence data well.

## Other regularization tools, connecting back to earlier lessons

- **L2 weight decay** (Lesson 022's Ridge, applied to network weights):
  `torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)`.
- **Early stopping** (Lesson 040) — stop before the model has a chance to
  overfit.
- **Data augmentation** (previewed here, detailed for images in Lesson 043) —
  synthetically expanding the effective training set.

## A practical mental model

Think of Dropout, BatchNorm, weight decay, and early stopping as different
knobs on the same bias-variance dial from Lesson 017 — each attacks
overfitting (variance) through a different mechanism (redundancy forcing,
activation stabilization, weight-magnitude penalty, training-duration
limiting respectively), and real architectures typically combine several of
them rather than relying on just one.
