# 03 — Solutions: PyTorch Fundamentals

## 1. Tensor basics

```python
import torch

a = torch.randn(3, 4)
b = a.reshape(4, 3)
c = a.reshape(2, 6)
print(a.numpy().shape)   # (3, 4), a real NumPy array
```

## 2. Matching Lesson 037/038's gradients

```python
x = torch.tensor(2.0)
w1 = torch.tensor(0.5, requires_grad=True)
b1 = torch.tensor(0.0, requires_grad=True)
w2 = torch.tensor(1.0, requires_grad=True)
b2 = torch.tensor(0.0, requires_grad=True)
y = torch.tensor(3.0)

z1 = w1 * x + b1
a1 = torch.relu(z1)
z2 = w2 * a1 + b2
loss = (z2 - y) ** 2
loss.backward()

print(w1.grad.item(), b1.grad.item(), w2.grad.item(), b2.grad.item())
# -8.0 -4.0 -4.0 -4.0 -- exact match to Lessons 037 and 038
```

## 3. Gradient accumulation bug, then fix

```python
for i in range(2):
    z1 = w1 * x + b1
    a1 = torch.relu(z1)
    z2 = w2 * a1 + b2
    loss = (z2 - y) ** 2
    loss.backward()
    print(f"iter {i}: w1.grad = {w1.grad.item()}")
    # without zeroing: iter 0 -> -8.0, iter 1 -> -16.0 (accumulated!)

# fixed version:
w1.grad = None
b1.grad = None
w2.grad = None
b2.grad = None
```

The second iteration's gradient is double the first (`-16.0` instead of
`-8.0`) because PyTorch tensors accumulate gradients via `+=` internally —
exactly like Lesson 038's `Value.grad += ...` — so without resetting,
`backward()` calls compound rather than replace.

## 4. XOR MLP in PyTorch

```python
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 4)
        self.fc2 = nn.Linear(4, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        return self.fc2(x)

X = torch.tensor([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
Y = torch.tensor([[-1.],[1.],[1.],[-1.]])

model = MLP()
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

for epoch in range(500):
    y_hat = model(X)
    loss = ((y_hat - Y) ** 2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(epoch, loss.item())

print(model(X).detach())
```

Loss should approach 0, mirroring Lesson 038's from-scratch result almost
exactly — same problem, same underlying math, now running through PyTorch's
engine.

## 5. SGD vs Adam

```python
for opt_name, opt_cls in [("SGD", torch.optim.SGD), ("Adam", torch.optim.Adam)]:
    torch.manual_seed(0)
    m = MLP()
    opt = opt_cls(m.parameters(), lr=0.1)
    for epoch in range(300):
        y_hat = m(X)
        loss = ((y_hat - Y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    print(opt_name, "final loss:", loss.item())
```

Adam often converges faster (lower loss in the same number of epochs) than
plain SGD at the same learning rate on small problems like this, previewing
Lesson 041's optimizer comparison in more depth.

## 6. CPU vs GPU timing

```python
import time

def bench(device):
    a = torch.randn(1000, 1000, device=device)
    b = torch.randn(1000, 1000, device=device)
    torch.cuda.synchronize() if device == "cuda" else None
    t0 = time.time()
    for _ in range(1000):
        c = a @ b
    torch.cuda.synchronize() if device == "cuda" else None
    return time.time() - t0

print("cpu:", bench("cpu"))
if torch.cuda.is_available():
    print("cuda:", bench("cuda"))
else:
    print("no GPU available in this environment")
```

On a machine with a GPU, expect a substantial speedup (often 10-50x or more
for matrix multiplication at this size) — the practical reason every
serious deep learning workload, especially LLM training (Module 11), runs
on GPU rather than CPU. `torch.cuda.synchronize()` is needed before timing
GPU code because CUDA operations are asynchronous by default — without it
you'd measure how long it took to *launch* the operations, not how long they
actually took to run.
