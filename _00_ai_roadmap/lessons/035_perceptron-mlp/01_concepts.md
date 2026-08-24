# 01 — Concepts: Perceptron & Multi-Layer Perceptron

## The perceptron: logistic regression's mechanical ancestor

```
z = w·x + b
output = 1 if z > 0 else 0
```

Compare to Lesson 023's logistic regression — same linear combination, but a
hard 0/1 step function instead of a smooth sigmoid. The original perceptron
learning rule adjusts weights whenever a prediction is wrong:
`w += lr * (true_label - prediction) * x`. It's guaranteed to converge *only*
if the data is linearly separable — and famously, a single perceptron
**cannot learn XOR** (not linearly separable), a limitation that stalled
neural network research for years until multi-layer networks were shown to
overcome it.

## Why stacking layers helps: composing simple functions into complex ones

A **Multi-Layer Perceptron (MLP)** stacks layers of these units:

```
h1 = activation(W1 @ x + b1)      # hidden layer
h2 = activation(W2 @ h1 + b2)     # another hidden layer
output = W3 @ h2 + b3              # output layer
```

Each layer is a linear transformation (Lesson 011) followed by a
**nonlinear activation function** — and that nonlinearity is essential:
without it, stacking linear layers just collapses into one big linear
transformation (`W2 @ (W1 @ x) = (W2 @ W1) @ x`, still linear), no more
powerful than a single layer, no matter how deep. Nonlinear activations
(Lesson 036) are what let deep networks represent genuinely complex,
curved decision boundaries — solving XOR, and everything harder.

## The universal approximation theorem (informally)

A feedforward network with even a single hidden layer (with enough units and
a suitable nonlinearity) can approximate *any* continuous function to
arbitrary precision. This is a big claim, but a subtle one: it says such a
network **exists**, not that gradient descent will *find* it, nor that a
shallow network is practical or efficient — depth (many layers) turns out to
be far more parameter-efficient in practice than making one hidden layer
arbitrarily wide, which is exactly why "deep" learning uses many layers
rather than one huge one.

## From forward pass to prediction

```python
import numpy as np

def relu(x):
    return np.maximum(0, x)

def forward(x, W1, b1, W2, b2):
    h = relu(x @ W1 + b1)
    out = h @ W2 + b2
    return out
```

This is exactly what Lesson 038 will train from scratch, and what Lesson 040
will build in PyTorch instead of by hand.

## Architecture choices you'll be making from here on

- **Width** (units per layer) and **depth** (number of layers): more
  capacity, more overfitting risk (Lesson 017) — tuned like any other
  hyperparameter.
- **Activation function** (Lesson 036): ReLU is the modern default for
  hidden layers.
- **Output layer shape**: 1 unit + sigmoid for binary classification
  (Lesson 023's exact setup), `n_classes` units + softmax for multi-class
  (Lesson 036), 1 unit with no activation for regression.

## XOR as the canonical "why depth matters" example

```
XOR truth table:
(0,0) -> 0
(0,1) -> 1
(1,0) -> 1
(1,1) -> 0
```

No single straight line can separate the 1s from the 0s here — genuinely not
linearly separable. A single perceptron/logistic regression provably cannot
solve it; a 2-layer MLP with just 2 hidden units and a nonlinearity can,
easily. You'll prove this to yourself directly in the practicals.
