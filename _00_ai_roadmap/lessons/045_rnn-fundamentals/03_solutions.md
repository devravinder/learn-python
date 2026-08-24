# 03 — Solutions: RNN Fundamentals

## 1. RNN cell from scratch

```python
import numpy as np

rng = np.random.default_rng(0)
input_size, hidden_size = 3, 4
W_xh = rng.normal(0, 0.3, (hidden_size, input_size))
W_hh = rng.normal(0, 0.3, (hidden_size, hidden_size))
b_h = np.zeros(hidden_size)

def rnn_cell(x_t, h_prev):
    return np.tanh(W_xh @ x_t + W_hh @ h_prev + b_h)

sequence = rng.normal(size=(5, input_size))
h = np.zeros(hidden_size)
for t, x_t in enumerate(sequence):
    h = rnn_cell(x_t, h)
    print(t, h)
```

## 2. Compare to `nn.RNN`

```python
import torch
import torch.nn as nn

torch_rnn = nn.RNN(input_size=3, hidden_size=4, batch_first=True, bias=True)
with torch.no_grad():
    torch_rnn.weight_ih_l0.copy_(torch.tensor(W_xh, dtype=torch.float32))
    torch_rnn.weight_hh_l0.copy_(torch.tensor(W_hh, dtype=torch.float32))
    torch_rnn.bias_ih_l0.zero_()
    torch_rnn.bias_hh_l0.zero_()

x = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)   # (1, 5, 3)
output, h_n = torch_rnn(x)
print(h_n.squeeze().detach().numpy())
print(h)   # should match closely
```

(PyTorch's `nn.RNN` splits the bias into `bias_ih` and `bias_hh` that sum
together internally — zeroing both replicates a single `b_h=0` from the
from-scratch version.)

## 3. Many-to-one sequence classifier

```python
class SumClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(1, 16, batch_first=True)
        self.fc = nn.Linear(16, 1)

    def forward(self, x):
        _, h_n = self.rnn(x)
        return self.fc(h_n.squeeze(0))

rng = np.random.default_rng(0)
X = rng.normal(size=(500, 10, 1)).astype(np.float32)
y = (X.sum(axis=1) > 0).astype(np.float32)

X_t, y_t = torch.tensor(X), torch.tensor(y)
model = SumClassifier()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
loss_fn = nn.BCEWithLogitsLoss()

for epoch in range(100):
    optimizer.zero_grad()
    loss = loss_fn(model(X_t), y_t)
    loss.backward()
    optimizer.step()

preds = (torch.sigmoid(model(X_t)) > 0.5).float()
print("accuracy:", (preds == y_t).float().mean().item())
```

Should reach high accuracy (often >90%) — summing a sequence is a
relatively easy task for an RNN to learn via its hidden state accumulation.

## 4. Vanishing gradients vs sequence length

```python
for seq_len in [10, 50, 100, 200]:
    torch.manual_seed(0)
    rnn = nn.RNN(1, 8, batch_first=True)
    x = torch.randn(1, seq_len, 1, requires_grad=True)
    output, h_n = rnn(x)
    loss = h_n.sum()
    loss.backward()
    print(seq_len, rnn.weight_hh_l0.grad.norm().item())
```

Gradient norm on `weight_hh_l0` (which is reused, and thus accumulates
contributions from, every timestep) often does **not** monotonically
vanish to exactly 0 with plain `tanh` RNNs at these lengths (it depends
heavily on initialization scale), but the *effective* long-range signal
(how much the loss actually depends on very early timesteps specifically)
degrades substantially with length — a subtler effect than the raw gradient
norm alone shows, which is part of why the vanishing gradient problem in
RNNs is often diagnosed via training difficulty on long-range-dependency
tasks rather than by watching a single gradient norm number.

## 5. Gradient clipping preventing instability

```python
torch.manual_seed(0)
model = SumClassifier()
optimizer = torch.optim.SGD(model.parameters(), lr=1.0)   # deliberately too large

for use_clipping in [False, True]:
    torch.manual_seed(0)
    model = SumClassifier()
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    diverged = False
    for epoch in range(50):
        optimizer.zero_grad()
        loss = loss_fn(model(X_t), y_t)
        loss.backward()
        if use_clipping:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if torch.isnan(loss):
            diverged = True
            break
    print("clipping:", use_clipping, "diverged:", diverged, "final loss:", loss.item())
```

Without clipping, the large learning rate often causes the loss to become
`nan` (exploded gradients) within a handful of epochs; with clipping,
training typically remains stable (loss stays finite, even if convergence
is slow at such a high nominal learning rate) — a direct demonstration of
what gradient clipping actually prevents.

## 6. Unidirectional vs bidirectional on a "needs future context" task

```python
def make_local_max_data(n_seqs=500, seq_len=10, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_seqs, seq_len, 1)).astype(np.float32)
    y = np.zeros((n_seqs, seq_len, 1), dtype=np.float32)
    for i in range(n_seqs):
        for t in range(1, seq_len - 1):
            if X[i, t, 0] > X[i, t-1, 0] and X[i, t, 0] > X[i, t+1, 0]:
                y[i, t, 0] = 1.0
    return X, y

X, y = make_local_max_data()
X_t, y_t = torch.tensor(X), torch.tensor(y)

class Tagger(nn.Module):
    def __init__(self, bidirectional):
        super().__init__()
        self.rnn = nn.RNN(1, 16, batch_first=True, bidirectional=bidirectional)
        in_dim = 32 if bidirectional else 16
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out)

for bidir in [False, True]:
    torch.manual_seed(0)
    model = Tagger(bidir)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    for epoch in range(200):
        optimizer.zero_grad()
        loss = loss_fn(model(X_t), y_t)
        loss.backward()
        optimizer.step()
    preds = (torch.sigmoid(model(X_t)) > 0.5).float()
    acc = (preds == y_t).float().mean().item()
    print("bidirectional:", bidir, "accuracy:", acc, "final loss:", loss.item())
```

The bidirectional model should noticeably outperform the unidirectional
one on this task, since detecting "is this a local maximum" fundamentally
requires knowing the *next* element too — information a unidirectional
(forward-only) RNN structurally cannot access at the time it processes
position `t`, no matter how well-trained.
