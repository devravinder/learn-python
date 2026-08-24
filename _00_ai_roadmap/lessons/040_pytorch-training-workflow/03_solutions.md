# 03 — Solutions: The PyTorch Training Workflow

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

X, y = make_moons(n_samples=1000, noise=0.2, random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)
```

## 1. Dataset + DataLoader

```python
class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(TabularDataset(X_train, y_train), batch_size=32, shuffle=True)
val_loader = DataLoader(TabularDataset(X_val, y_val), batch_size=32)

xb, yb = next(iter(train_loader))
print(xb.shape, yb.shape)   # torch.Size([32, 2]) torch.Size([32, 1])
```

## 2–3. Training with tracked loss curves

```python
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16), nn.ReLU(),
            nn.Linear(16, 16), nn.ReLU(),
            nn.Linear(16, 1),
        )
    def forward(self, x):
        return self.net(x)

model = MLP()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.BCEWithLogitsLoss()

train_losses, val_losses = [], []
for epoch in range(50):
    model.train()
    epoch_loss = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    train_losses.append(epoch_loss / len(train_loader))

    model.eval()
    with torch.no_grad():
        val_loss = sum(loss_fn(model(xb), yb).item() for xb, yb in val_loader) / len(val_loader)
    val_losses.append(val_loss)

import matplotlib.pyplot as plt
plt.plot(train_losses, label="train")
plt.plot(val_losses, label="val")
plt.legend()
plt.show()
```

On this moderately-sized, moderately-noisy dataset, train and validation
loss typically track closely together for the full 50 epochs (the model
isn't overparameterized relative to the data) — a case where you would
*not* see Lesson 017's overfitting divergence; try a much larger hidden
size (e.g. 256, 256) or far fewer training samples to reproduce the
divergence pattern deliberately.

## 4. Early stopping

```python
model = MLP()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
best_val_loss = float("inf")
patience, patience_counter = 5, 0

for epoch in range(200):
    model.train()
    for xb, yb in train_loader:
        optimizer.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_loss = sum(loss_fn(model(xb), yb).item() for xb, yb in val_loader) / len(val_loader)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "best_model.pt")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"early stopping at epoch {epoch}")
            break
```

## 5. Save/reload and verify

```python
reloaded = MLP()
reloaded.load_state_dict(torch.load("best_model.pt"))
reloaded.eval()

with torch.no_grad():
    val_acc = sum(
        ((torch.sigmoid(reloaded(xb)) > 0.5).float() == yb).float().mean().item()
        for xb, yb in val_loader
    ) / len(val_loader)
print("reloaded val accuracy:", val_acc)
```

Note `torch.sigmoid(...)` is applied manually here at inference time, since
`BCEWithLogitsLoss` expects raw logits during training and doesn't apply
sigmoid for you outside of the loss computation itself.

## 6. Loss/output shape mismatch

```python
# broken: CrossEntropyLoss expects one logit PER CLASS
model_binary_wrong = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1))
try:
    out = model_binary_wrong(torch.tensor(X_train[:4], dtype=torch.float32))
    loss = nn.CrossEntropyLoss()(out, torch.tensor(y_train[:4], dtype=torch.long))
except Exception as e:
    print("error:", e)

# fix option A: 2 output units + CrossEntropyLoss + integer labels
model_a = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 2))
out_a = model_a(torch.tensor(X_train[:4], dtype=torch.float32))
loss_a = nn.CrossEntropyLoss()(out_a, torch.tensor(y_train[:4], dtype=torch.long))

# fix option B: 1 output unit + BCEWithLogitsLoss + float 0/1 labels
model_b = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1))
out_b = model_b(torch.tensor(X_train[:4], dtype=torch.float32))
loss_b = nn.BCEWithLogitsLoss()(out_b, torch.tensor(y_train[:4], dtype=torch.float32).unsqueeze(1))
```

`CrossEntropyLoss` internally expects `(batch, num_classes)` logits and
integer class-index targets `(batch,)`; feeding it a single-logit output
raises a shape/target-type error. The two valid fixes are architecturally
different (`num_classes` output units + integer labels vs 1 output unit +
float 0/1 labels) — mixing conventions between model output shape and loss
function is one of the most common real PyTorch bugs, worth deliberately
triggering once so the error message is recognizable later.
