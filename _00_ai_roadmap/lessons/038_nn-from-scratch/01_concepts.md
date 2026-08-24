# 01 — Concepts: Neural Network From Scratch

## The idea: wrap every number in a node that remembers how it was made

Lesson 037 hand-derived backprop for one tiny network. To make this
automatic for *any* expression, wrap every scalar in a `Value` object that
tracks: its data, its gradient (initially 0), which operation produced it,
and which `Value`s were its inputs. Every operation (`+`, `*`, `**`, `relu`)
also stores a tiny `_backward` closure that knows *that operation's* local
derivative (Lesson 013's rules) and how to add its contribution to its
inputs' gradients.

```python
class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")
        def _backward():
            self.grad += out.grad     # d(out)/d(self) = 1
            other.grad += out.grad    # d(out)/d(other) = 1
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")
        def _backward():
            self.grad += other.data * out.grad   # d(out)/d(self) = other.data
            other.grad += self.data * out.grad   # d(out)/d(other) = self.data
        out._backward = _backward
        return out
```

Notice `+=`, not `=`, in every `_backward` — this is exactly Lesson 037's
"sum gradients at a branch" rule, applied automatically: if a `Value` is
used in multiple places, each use contributes its own gradient, and they
accumulate.

## Backward pass: topological sort, then apply in reverse

To call every node's `_backward()` in the correct order (each node's
gradient must be fully accumulated *before* it propagates further back), do
a topological sort of the computation graph, then walk it in reverse:

```python
def backward(self):
    topo = []
    visited = set()
    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build_topo(child)
            topo.append(v)
    build_topo(self)

    self.grad = 1.0   # dL/dL = 1, the seed gradient
    for v in reversed(topo):
        v._backward()
```

Calling `loss.backward()` now computes gradients for **every** `Value` that
contributed to `loss`, however deep the graph — this is a complete,
general-purpose autograd engine in under 40 lines.

## Building a neuron, a layer, and an MLP on top

```python
import random

class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0.0)

    def __call__(self, x):
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        return act.relu()

    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

class MLP:
    def __init__(self, nin, nouts):
        sizes = [nin] + nouts
        self.layers = [Layer(sizes[i], sizes[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
```

This is a **complete neural network implementation** — forward pass, and
(via the `Value` engine) backward pass — with zero dependencies beyond
Python's standard library.

## The training loop looks exactly like Lesson 015's gradient descent

```python
model = MLP(2, [4, 4, 1])   # 2 inputs -> 4 -> 4 -> 1 output

for epoch in range(100):
    # forward pass on the whole dataset
    y_pred = [model(x) for x in X]
    loss = sum((yp - yt)**2 for yp, yt in zip(y_pred, y)) * (1.0 / len(y))

    # zero gradients (they accumulate via += , so must reset each step)
    for p in model.parameters():
        p.grad = 0.0

    loss.backward()

    # gradient descent update
    for p in model.parameters():
        p.data -= 0.05 * p.grad
```

**Every piece of this — the `Value` engine, `Neuron`/`Layer`/`MLP`, and this
training loop — is literally what PyTorch does internally**, just with
tensors (batched arrays) instead of individual scalars, and heavily
optimized in C++/CUDA. Understanding this scalar version fully is what makes
Lesson 039's `tensor.backward()` feel like a natural continuation instead of
a magic trick.
