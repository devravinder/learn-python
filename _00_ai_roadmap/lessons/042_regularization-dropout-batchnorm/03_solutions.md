# 03 — Solutions: Dropout & BatchNorm

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

X, y = make_moons(n_samples=200, noise=0.3, random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=0)
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_val_t = torch.tensor(X_val, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

def train(model, epochs=300, weight_decay=0):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    train_losses, val_losses = [], []
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = loss_fn(model(X_train_t), y_train_t)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_losses.append(loss_fn(model(X_val_t), y_val_t).item())
    return train_losses, val_losses
```

## 1. Overfitting baseline

```python
baseline = nn.Sequential(
    nn.Linear(2, 256), nn.ReLU(),
    nn.Linear(256, 256), nn.ReLU(),
    nn.Linear(256, 1),
)
train_losses, val_losses = train(baseline)
plt.plot(train_losses, label="train"); plt.plot(val_losses, label="val"); plt.legend(); plt.show()
```

With 200 training samples and a 256-256 network, expect training loss to
approach 0 while validation loss plateaus or creeps upward after some
point — Lesson 017's overfitting signature, deliberately induced here by
using far more capacity than the data needs.

## 2. Dropout

```python
dropout_model = nn.Sequential(
    nn.Linear(2, 256), nn.ReLU(), nn.Dropout(0.5),
    nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.5),
    nn.Linear(256, 1),
)
train_losses_d, val_losses_d = train(dropout_model)
plt.plot(train_losses_d, label="train"); plt.plot(val_losses_d, label="val"); plt.legend(); plt.show()
```

The train/val gap should narrow compared to Q1 — training loss decreases
more slowly (dropout makes the effective network weaker/noisier per step)
but validation loss should track more closely to it, indicating better
generalization.

## 3. BatchNorm and convergence speed

```python
bn_model = nn.Sequential(
    nn.Linear(2, 256), nn.BatchNorm1d(256), nn.ReLU(),
    nn.Linear(256, 256), nn.BatchNorm1d(256), nn.ReLU(),
    nn.Linear(256, 1),
)
train_losses_bn, val_losses_bn = train(bn_model)

target = 0.1
epochs_to_target_baseline = next((i for i, l in enumerate(train_losses) if l < target), None)
epochs_to_target_bn = next((i for i, l in enumerate(train_losses_bn) if l < target), None)
print("epochs to reach loss<0.1 -- baseline:", epochs_to_target_baseline, "batchnorm:", epochs_to_target_bn)
```

BatchNorm often reaches a given training loss threshold in fewer epochs
than the unregularized baseline, consistent with its stabilizing effect on
layer input distributions allowing faster, more reliable convergence.

## 4. Dropout train/eval mismatch

```python
dropout_model.train()
with torch.no_grad():
    pred1 = dropout_model(X_val_t[:1])
    pred2 = dropout_model(X_val_t[:1])
print("train mode, same input twice:", pred1.item(), pred2.item())   # likely different

dropout_model.eval()
with torch.no_grad():
    pred3 = dropout_model(X_val_t[:1])
    pred4 = dropout_model(X_val_t[:1])
print("eval mode, same input twice:", pred3.item(), pred4.item())    # identical
```

In `.train()` mode, dropout randomly zeroes different units each call,
producing different outputs for the identical input; `.eval()` disables
this randomness, making inference deterministic — exactly why forgetting
`model.eval()` before evaluation/deployment is a real, easy-to-make bug.

## 5. Weight decay

```python
wd_model = nn.Sequential(
    nn.Linear(2, 256), nn.ReLU(),
    nn.Linear(256, 256), nn.ReLU(),
    nn.Linear(256, 1),
)
train_losses_wd, val_losses_wd = train(wd_model, weight_decay=1e-3)
plt.plot(train_losses_wd, label="train"); plt.plot(val_losses_wd, label="val"); plt.legend(); plt.show()
```

Weight decay should also narrow the train/val gap compared to Q1, via a
different mechanism (penalizing large weights directly, Lesson 022) than
dropout's redundancy-forcing.

## 6. Combining all three

```python
combined_model = nn.Sequential(
    nn.Linear(2, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
    nn.Linear(256, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
    nn.Linear(256, 1),
)
train_losses_c, val_losses_c = train(combined_model, weight_decay=1e-4)
print("final val loss - baseline:", val_losses[-1], "combined:", val_losses_c[-1])
```

Combining all three often gives the best (lowest) final validation loss,
but not always by a large margin over the single best individual technique
— and over-combining (e.g. very high dropout *and* strong weight decay
*and* small BatchNorm batches) can sometimes under-fit by over-constraining
the model. The practical lesson: try one regularization technique at a
time, measure its effect on the train/val gap, and only stack more if the
gap is still too wide — don't reach for every regularizer at once by
default.
