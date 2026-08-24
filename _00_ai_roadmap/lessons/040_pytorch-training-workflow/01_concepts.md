# 01 — Concepts: The PyTorch Training Workflow

## `Dataset` and `DataLoader` — the mini-batch machinery from Lesson 015

Recall Lesson 002's `Dataset` class exercise (`__len__` + `__getitem__`) —
this is exactly what PyTorch expects:

```python
from torch.utils.data import Dataset, DataLoader

class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(TabularDataset(X_train, y_train), batch_size=32, shuffle=True)
```

`DataLoader` handles Lesson 015's mini-batch gradient descent mechanics for
you: shuffling each epoch, batching, and (with `num_workers > 0`) loading
data in parallel — essential once datasets are too large to fit a single
`X`/`y` tensor conveniently in memory, and standard practice even when they
do fit.

## The full training loop

```python
model = MLP(...)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()   # or nn.CrossEntropyLoss() for classification

for epoch in range(num_epochs):
    model.train()                        # enables dropout/batchnorm training behavior (Lesson 042)
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        loss.backward()
        optimizer.step()

    model.eval()                          # disables dropout/batchnorm training behavior
    with torch.no_grad():
        val_loss = sum(loss_fn(model(xb), yb).item() for xb, yb in val_loader) / len(val_loader)
    print(f"epoch {epoch}: val_loss={val_loss:.4f}")
```

`model.train()`/`model.eval()` matter starting Lesson 042 (dropout behaves
differently at train vs eval time) — good habit to include from the start.

## Loss functions: matching the right one to your problem

```python
nn.MSELoss()            # regression
nn.BCEWithLogitsLoss()  # binary classification - takes raw logits, applies sigmoid internally
nn.CrossEntropyLoss()   # multi-class classification - takes raw logits, applies softmax internally
```

Note: `CrossEntropyLoss` and `BCEWithLogitsLoss` expect **raw logits**, not
post-sigmoid/softmax probabilities — applying softmax yourself *and* using
these losses double-applies it, a common bug. This combined
"logits + built-in activation + loss" design also happens to be more
numerically stable (Lesson 007's max-subtraction trick is applied
internally).

## Early stopping — the practical bias-variance control (Lesson 017)

```python
best_val_loss = float("inf")
patience_counter = 0
patience = 10

for epoch in range(num_epochs):
    ... # train + compute val_loss as above
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "best_model.pt")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("early stopping")
            break
```

Stop training once validation loss stops improving for `patience` epochs —
directly preventing the overfitting regime from Lesson 017, where training
loss keeps dropping while validation loss gets worse.

## Saving and loading models

```python
torch.save(model.state_dict(), "model.pt")           # save just the weights (recommended)

model = MLP(...)                                        # recreate the architecture
model.load_state_dict(torch.load("model.pt"))
model.eval()
```

Saving `state_dict()` (a dict of tensor weights) rather than the whole
object is the standard, more portable approach — you need the model class
definition available to reload it, but avoid pickling issues tied to exact
code versions.

## Reproducibility

```python
torch.manual_seed(0)
```

Sets PyTorch's random seed for weight initialization and any random
operations (dropout, data shuffling) — necessary for comparing runs fairly
(e.g. Lesson 039's optimizer comparison), though full bit-for-bit
reproducibility on GPU requires additional settings due to some
non-deterministic parallel operations.
