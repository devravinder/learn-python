# 02 — Practicals: The PyTorch Training Workflow

Use a synthetic binary classification dataset:

```python
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

X, y = make_moons(n_samples=1000, noise=0.2, random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)
```

1. Implement the `TabularDataset` class from `01_concepts.md` and wrap
   `X_train, y_train` in a `DataLoader` with `batch_size=32, shuffle=True`.
   Iterate one batch and print its shape.

2. Build an MLP (`nn.Module`, 2 -> 16 -> 16 -> 1) with ReLU hidden
   activations and a raw logit output (no sigmoid). Train it using
   `nn.BCEWithLogitsLoss()` and `torch.optim.Adam(lr=1e-3)` for 50 epochs,
   tracking both training loss and validation loss (on `X_val, y_val`) each
   epoch.

3. Plot training loss and validation loss curves on the same chart. Do they
   track together, or does validation loss start rising while training loss
   keeps falling (Lesson 017's overfitting signature)?

4. Implement early stopping (per `01_concepts.md`) with `patience=5`. Retrain
   from scratch and report at which epoch it actually stopped.

5. Save the best model's `state_dict()`. In a fresh Python session (or just
   a fresh model instance), reload it and confirm its validation accuracy
   matches what you saw during training.

6. Deliberately swap in `nn.CrossEntropyLoss()` with a single-logit output
   (mismatched shapes/expectations — `CrossEntropyLoss` expects one logit
   *per class*, not one logit for binary). Observe the error PyTorch raises,
   then fix it properly: either use 2 output units with
   `nn.CrossEntropyLoss()` and integer labels, or 1 output unit with
   `nn.BCEWithLogitsLoss()` and float 0/1 labels. Note in a comment which
   combination you used and why they must match.
