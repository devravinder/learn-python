# 01 — Concepts: PyTorch Fundamentals

## Tensors: NumPy arrays that know how to compute gradients

A `torch.Tensor` is functionally like a NumPy `ndarray` (Lesson 003) — same
shape/dtype/broadcasting rules — plus an optional `requires_grad` flag that
turns on Lesson 038's exact mechanism, engineered in C++/CUDA for speed and
n-dimensional arrays instead of scalars.

```python
import torch

x = torch.tensor([1.0, 2.0, 3.0])
y = torch.from_numpy(np.array([1.0, 2.0, 3.0]))   # convert from NumPy
z = x.numpy()                                       # convert back

a = torch.zeros(3, 4)
b = torch.randn(2, 3)          # standard normal, like np.random.default_rng().normal
c = torch.arange(0, 10, 2)
```

Nearly every NumPy operation you learned in Lesson 003 has a direct PyTorch
equivalent: `@` for matmul, broadcasting rules identical, `.reshape()`,
`.sum(axis=...)` (PyTorch calls it `dim` instead of `axis`), etc.

## Autograd: Lesson 038's engine, at tensor scale

```python
w = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)
x = torch.tensor(3.0)
y = torch.tensor(10.0)

y_hat = w * x + b
loss = (y_hat - y) ** 2

loss.backward()          # exactly your Value.backward() from Lesson 038
print(w.grad, b.grad)    # gradients computed automatically
```

This is **the same computation graph + topological backward pass** you
implemented by hand — `requires_grad=True` marks a tensor as a graph leaf to
track, and every operation on tracked tensors builds the graph
automatically, just like your `Value` class did implicitly through
`__add__`/`__mul__`.

## Gradients accumulate — you must zero them (a direct callback to Lesson 038)

```python
w.grad = None   # or optimizer.zero_grad() when using an optimizer (Lesson 041)
```

Exactly the `p.grad = 0.0` reset you had to remember in Lesson 038's
training loop — forgetting this is one of the most common PyTorch bugs, and
now you know precisely *why* it's necessary (gradients add via `+=`
internally, same as your own engine).

## `nn.Module`: organizing parameters like your `Neuron`/`Layer`/`MLP`

```python
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, n_in, n_hidden, n_out):
        super().__init__()
        self.fc1 = nn.Linear(n_in, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_out)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

model = MLP(2, 16, 1)
print(list(model.parameters()))   # every nn.Linear's weights+biases, auto-registered
```

`nn.Linear(n_in, n_out)` is a full `W @ x + b` layer with its weights
already set up as `requires_grad=True` tensors — directly analogous to your
`Layer` class, just vectorized and with better-tuned initialization built
in.

## Training loop shape (should look extremely familiar)

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(100):
    y_hat = model(X)
    loss = ((y_hat - y) ** 2).mean()

    optimizer.zero_grad()   # your `p.grad = 0.0` loop, one call
    loss.backward()         # your `loss.backward()`, unchanged
    optimizer.step()        # your `p.data -= lr * p.grad` loop, one call
```

**This is exactly Lesson 038's training loop**, with three lines replacing
three manual loops. If this doesn't feel new, that's the point — Lesson 038
was designed so this would feel obvious rather than magic.

## GPU acceleration — the same code, one call different

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
X = X.to(device)
```

Moving tensors/models to a GPU is where PyTorch's real practical advantage
over a hand-written engine shows up: the same operations run vastly faster
on GPU hardware, essential once you get to CNNs (Lesson 043) and especially
training your own LLM (Module 11), which is computationally infeasible on
CPU alone for anything beyond a toy scale.

## `torch.no_grad()` — turning off tracking when you don't need it

```python
with torch.no_grad():
    predictions = model(X_test)   # inference: no need to build a graph or track gradients
```

Saves memory and computation whenever you're not going to call
`.backward()` — every evaluation/inference pass should use this.
