# 02 — Practicals: Dropout & BatchNorm

Use a small, overfit-prone setup on purpose (few samples, big network):

```python
import torch
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

X, y = make_moons(n_samples=200, noise=0.3, random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=0)
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_val_t = torch.tensor(X_val, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
```

1. Build a deliberately oversized MLP (2 -> 256 -> 256 -> 1, ReLU, no
   regularization). Train for 300 epochs (full-batch, `Adam(lr=1e-3)`),
   tracking train/val loss. Confirm you see Lesson 017's overfitting
   signature (val loss rising or plateauing while train loss keeps
   dropping).

2. Add `nn.Dropout(p=0.5)` after each hidden layer's activation. Retrain
   identically and compare the train/val loss curves to Q1 — does the gap
   shrink?

3. Instead (remove dropout), add `nn.BatchNorm1d` after each `Linear`
   (before the activation). Retrain and compare convergence *speed* (how
   many epochs to reach a fixed low training loss) to Q1 — does BatchNorm
   speed up training as `01_concepts.md` suggests?

4. Demonstrate the train/eval dropout bug directly: build the Q2 model,
   train it, then run inference **without** calling `model.eval()` first
   (leave it in `.train()` mode) on the same input twice. Show the two
   predictions differ (dropout still randomly active) — then call
   `model.eval()` and show predictions become deterministic.

5. Add `weight_decay=1e-3` to the optimizer on the Q1 (unregularized
   architecture) model instead of dropout/batchnorm. Compare the train/val
   gap to Q1's un-regularized baseline.

6. Combine Dropout + BatchNorm + weight_decay all together on the oversized
   network. Compare final validation loss to each individual technique
   alone (Q2, Q3, Q5) — does combining help, hurt, or make little
   difference here?
