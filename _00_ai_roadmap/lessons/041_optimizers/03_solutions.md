# 03 — Solutions: Optimizers

```python
import torch
import matplotlib.pyplot as plt

def rosenbrock(xy):
    x, y = xy[0], xy[1]
    return (1 - x)**2 + 100 * (y - x**2)**2

def optimize(opt_cls, opt_kwargs, steps=200):
    xy = torch.tensor([-1.5, 2.0], requires_grad=True)
    optimizer = opt_cls([xy], **opt_kwargs)
    losses = []
    for _ in range(steps):
        optimizer.zero_grad()
        loss = rosenbrock(xy)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses, xy.detach().clone()
```

## 1–3. Comparing optimizers on Rosenbrock

```python
sgd_losses, sgd_final = optimize(torch.optim.SGD, {"lr": 0.001})
momentum_losses, momentum_final = optimize(torch.optim.SGD, {"lr": 0.001, "momentum": 0.9})
adam_losses, adam_final = optimize(torch.optim.Adam, {"lr": 0.05})

print("SGD final:", sgd_losses[-1], sgd_final)
print("Momentum final:", momentum_losses[-1], momentum_final)
print("Adam final:", adam_losses[-1], adam_final)
```

Plain SGD typically makes very slow progress on Rosenbrock's narrow curved
valley within 200 steps at this small learning rate (still far from
`(1,1)`); momentum usually makes noticeably more progress by building up
speed along the valley's general direction; Adam often converges fastest of
the three here, adapting its effective step size per-dimension to the very
different curvatures along the valley's floor vs walls.

## 4. Loss curve comparison

```python
plt.plot(sgd_losses, label="SGD")
plt.plot(momentum_losses, label="SGD+momentum")
plt.plot(adam_losses, label="Adam")
plt.yscale("log")
plt.legend()
plt.show()
```

## 5. Optimizer comparison on XOR MLP

```python
import torch.nn as nn

X = torch.tensor([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
Y = torch.tensor([[-1.],[1.],[1.],[-1.]])

def make_model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 1))

configs = [
    ("SGD", torch.optim.SGD, {"lr": 0.1}),
    ("SGD+momentum", torch.optim.SGD, {"lr": 0.1, "momentum": 0.9}),
    ("RMSprop", torch.optim.RMSprop, {"lr": 0.01}),
    ("Adam", torch.optim.Adam, {"lr": 0.01}),
]

results = {}
for name, opt_cls, kwargs in configs:
    model = make_model()
    optimizer = opt_cls(model.parameters(), **kwargs)
    for epoch in range(100):
        optimizer.zero_grad()
        loss = ((model(X) - Y) ** 2).mean()
        loss.backward()
        optimizer.step()
    results[name] = loss.item()

for name, final_loss in results.items():
    print(f"{name:15s} final loss: {final_loss:.6f}")
```

Adam and RMSprop typically reach the lowest loss within 100 epochs on this
small problem; plain SGD usually lags noticeably behind at the same nominal
learning rate, consistent with Adam/RMSprop's adaptive per-parameter scaling
giving them an advantage on problems with uneven gradient magnitudes across
parameters.

## 6. Cosine annealing schedule

```python
model = make_model()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

lrs = []
for epoch in range(100):
    optimizer.zero_grad()
    loss = ((model(X) - Y) ** 2).mean()
    loss.backward()
    optimizer.step()
    scheduler.step()
    lrs.append(scheduler.get_last_lr()[0])

plt.plot(lrs)
plt.ylabel("learning rate")
plt.show()
print("final loss with schedule:", loss.item())
```

The learning rate plot should show a smooth cosine-shaped decay from 0.01
down toward 0 over the 100 epochs. On a problem this easy, final loss with
vs without the schedule is often similar (both converge fully); the
schedule's practical benefit shows up more clearly on harder, longer
training runs (like LLM pretraining, Module 11) where a decaying learning
rate helps fine-tune convergence in later stages without the instability a
constant high learning rate would cause.
