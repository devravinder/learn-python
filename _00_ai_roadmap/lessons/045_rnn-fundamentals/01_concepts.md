# 01 — Concepts: RNN Fundamentals

## Why not just feed a whole sequence into an MLP?

Sequences have variable length, and an MLP needs a fixed input size. More
fundamentally, an MLP has no notion of *order* — flattening a sentence into
a feature vector loses the sequential relationships (Lesson 010's vectors
have no inherent "before/after"). RNNs process a sequence one element at a
time, maintaining a **hidden state** that carries information forward.

## The recurrence

```
h_t = tanh(W_xh @ x_t + W_hh @ h_(t-1) + b_h)
y_t = W_hy @ h_t + b_y
```

At each timestep `t`: combine the current input `x_t` with the *previous*
hidden state `h_(t-1)` to produce a new hidden state `h_t`. The **same
weights** (`W_xh, W_hh, W_hy`) are reused at every timestep — a direct
parallel to CNN's parameter sharing (Lesson 043), here shared across time
instead of across spatial position.

```python
import torch.nn as nn
rnn = nn.RNN(input_size=10, hidden_size=20, batch_first=True)
output, h_n = rnn(x)   # x: (batch, seq_len, input_size)
```

## Unrolling through time — backprop's new setting

To train an RNN, "unroll" it into a chain of identical layers (one per
timestep), then apply ordinary backpropagation (Lesson 037) through this
unrolled graph — called **Backpropagation Through Time (BPTT)**. Since the
same weights are reused at every timestep, gradients from every timestep's
contribution get summed (Lesson 037's "branching" rule, since the shared
weight is used — and contributes error — at every single timestep).

## The vanishing/exploding gradient problem, at a new scale

Gradients flowing back through many timesteps get multiplied by the same
weight matrix repeatedly (once per timestep) — analogous to Lesson 036's
sigmoid-chained-10-times example, but now potentially chained over
**hundreds** of timesteps for long sequences. If the weight matrix's
"effective magnitude" is <1, gradients shrink toward 0 (vanishing — the
network can't learn long-range dependencies, effectively "forgetting"
anything from many steps back); if >1, they grow toward infinity
(exploding — training becomes unstable). This is the core limitation that
motivates LSTM/GRU (Lesson 046).

**Gradient clipping** is a direct, simple mitigation for exploding
gradients:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

(Called right before `optimizer.step()` — caps the gradient's total norm,
preventing a single huge update from destabilizing training. Standard
practice for RNNs, and still used in Transformer/LLM training too.)

## RNN variants by input/output shape

- **Many-to-one**: sequence in, single output (e.g. sentiment classification
  of a whole sentence — use the final hidden state).
- **Many-to-many (aligned)**: one output per input timestep (e.g.
  part-of-speech tagging — every word gets a label).
- **Many-to-many (unaligned)**: encoder consumes the whole input, decoder
  generates a different-length output (e.g. translation — the direct setup
  for Lesson 047's Seq2Seq).
- **One-to-many**: single input, sequence out (e.g. image captioning —
  less common as a pure RNN task today).

## Bidirectional RNNs

Process the sequence both forward and backward, concatenating both hidden
states at each position — useful when the *whole* sequence is available
upfront (not for real-time generation, where future tokens don't exist yet)
and you want each position's representation informed by both past and
future context, e.g. for classification or tagging tasks.

```python
rnn = nn.RNN(input_size=10, hidden_size=20, bidirectional=True, batch_first=True)
```

## Where RNNs fit in this curriculum's arc

RNNs were the standard approach to sequence modeling (including early
machine translation and language models) before 2017's "Attention is All
You Need" paper (Lesson 058-060) showed that attention-based Transformers
handle long-range dependencies far better (no sequential bottleneck, no
vanishing gradient over many steps) *and* parallelize across the whole
sequence during training (RNNs are inherently sequential — timestep `t`
needs `h_(t-1)`, so you can't parallelize across time the way CNNs
parallelize across space). This lesson's limitations are exactly the
motivation for Lesson 058's attention mechanism.
