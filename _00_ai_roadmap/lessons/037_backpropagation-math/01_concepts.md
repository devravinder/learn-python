# 01 — Concepts: Backpropagation

## The problem backprop solves

To train a network with gradient descent (Lesson 015), you need
`∂Loss/∂w` for **every** weight in **every** layer. A network is a deep
composition of functions — exactly the setting for the multivariable chain
rule (Lesson 014). Backpropagation is just the chain rule, applied
systematically and efficiently, computing gradients for an entire network in
one backward pass instead of one painful derivation per weight.

## Computation graphs

Any expression can be drawn as a graph of operations. Example: `L = (w*x + b - y)^2` :

```
w ─┐
   ×──┐
x ─┘  │
      +──┐
b ─────┘ │
         ─── (subtract y) ──── ^2 ──> L
y ───────┘
```

Forward pass: compute left-to-right, storing every intermediate value.
Backward pass: apply the chain rule right-to-left, computing
`∂L/∂(each node)` using the *already-computed* gradient of whatever it feeds
into — each node only needs to know its own local derivative and the
gradient flowing back from downstream. This local-computation property is
what makes backprop efficient: **the cost of computing all gradients is
about the same as one extra forward pass**, regardless of how many
parameters there are.

## Worked example: full derivation for a tiny 2-layer network

```
x -> z1 = w1*x + b1 -> a1 = relu(z1) -> z2 = w2*a1 + b2 -> ŷ = z2
Loss L = (ŷ - y)^2
```

**Forward pass** (pick numbers): `x=2, w1=0.5, b1=0, w2=1.0, b2=0, y=3`

```
z1 = 0.5*2 + 0 = 1.0
a1 = relu(1.0) = 1.0
z2 = 1.0*1.0 + 0 = 1.0
ŷ = 1.0
L = (1.0 - 3)^2 = 4.0
```

**Backward pass** — apply the chain rule, one step at a time:

```
∂L/∂ŷ = 2*(ŷ - y) = 2*(1.0-3) = -4.0

∂L/∂z2 = ∂L/∂ŷ * ∂ŷ/∂z2 = -4.0 * 1 = -4.0        (ŷ = z2, so ∂ŷ/∂z2=1)

∂L/∂w2 = ∂L/∂z2 * ∂z2/∂w2 = -4.0 * a1 = -4.0 * 1.0 = -4.0
∂L/∂b2 = ∂L/∂z2 * ∂z2/∂b2 = -4.0 * 1 = -4.0
∂L/∂a1 = ∂L/∂z2 * ∂z2/∂a1 = -4.0 * w2 = -4.0 * 1.0 = -4.0

∂L/∂z1 = ∂L/∂a1 * ∂a1/∂z1 = -4.0 * relu'(1.0) = -4.0 * 1 = -4.0   (relu'(x>0)=1)

∂L/∂w1 = ∂L/∂z1 * ∂z1/∂w1 = -4.0 * x = -4.0 * 2 = -8.0
∂L/∂b1 = ∂L/∂z1 * ∂z1/∂b1 = -4.0 * 1 = -4.0
```

Every gradient reuses the *previous* one (`∂L/∂z2` reused for both `w2`'s
and `a1`'s gradients; `∂L/∂z1` reused for both `w1`'s and `b1`'s) — this
reuse, flowing backward through the graph exactly once, is the entire
efficiency trick.

## Why it's called "backpropagation"

The error signal (`∂L/∂(something)`) literally propagates backward through
the network, layer by layer, each layer passing back the gradient with
respect to *its own input* (which is the previous layer's output) — this is
why the multivariable chain rule (Lesson 014) is doing all the real work,
and "backprop" is just the name for applying it in this particular
efficient, graph-shaped order.

## What happens at a branch (a value used in multiple places)

If a value feeds into two different downstream computations, the
multivariable chain rule (Lesson 014) says to **sum** the gradients flowing
back from each path:

```
∂L/∂x = ∂L/∂path1 * ∂path1/∂x + ∂L/∂path2 * ∂path2/∂x
```

This matters directly for real networks: any weight or activation reused
across multiple positions (e.g. in an RNN, Lesson 045, the same weights are
reused at every timestep) accumulates gradient contributions from every
place it was used.

## Vectorized backprop (what you'll actually implement in Lesson 038)

Real layers process a whole batch of vectors at once via matrix
multiplication (Lesson 011), so the "local derivative" at each layer is a
Jacobian (Lesson 014), and the chain rule becomes a matrix multiplication
too. For a linear layer `z = x @ W + b`:

```
∂L/∂W = x.T @ (∂L/∂z)
∂L/∂b = sum(∂L/∂z, axis=0)
∂L/∂x = (∂L/∂z) @ W.T
```

`∂L/∂x` is what gets passed *backward* to the previous layer — the recursive
structure that makes deep backprop work layer-by-layer without ever needing
one giant symbolic derivative.
