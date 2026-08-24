# 01 — Concepts: LSTM & GRU

## The key idea: a separate memory highway, controlled by gates

Plain RNNs (Lesson 045) force *all* information through the same repeated
`tanh` transformation every timestep — nothing has a way to pass through
unchanged, which is exactly why gradients (and information) decay over long
sequences. LSTM adds a **cell state** `C_t` that can flow through timesteps
with only *additive*, gated modifications — much friendlier to gradient
flow (a direct conceptual cousin of ResNet's skip connection, Lesson 044:
both give gradients a path that doesn't have to pass through a squashing
nonlinearity at every step).

## LSTM's three gates (each a sigmoid, output in [0,1] — "how much to let through")

```
forget_gate  f_t = sigmoid(W_f @ [h_(t-1), x_t] + b_f)
input_gate   i_t = sigmoid(W_i @ [h_(t-1), x_t] + b_i)
output_gate  o_t = sigmoid(W_o @ [h_(t-1), x_t] + b_o)

candidate    C~_t = tanh(W_c @ [h_(t-1), x_t] + b_c)

cell state   C_t = f_t * C_(t-1) + i_t * C~_t     # <- the memory highway
hidden state h_t = o_t * tanh(C_t)
```

- **Forget gate**: how much of the *old* cell state to keep (`0`=forget
  everything, `1`=keep everything).
- **Input gate**: how much of the *new* candidate information to write in.
- **Output gate**: how much of the cell state to expose as this timestep's
  hidden state/output.

The cell state update `C_t = f_t * C_(t-1) + i_t * C~_t` is **additive** —
unlike a plain RNN's fully-multiplicative-through-tanh update, information
can pass through many timesteps nearly unchanged if `f_t ≈ 1` and
`i_t ≈ 0`, giving gradients a much more direct path backward and largely
solving the vanishing gradient problem for realistic sequence lengths (very
extreme lengths can still be challenging, which is part of why attention/
Transformers eventually became the dominant approach, Lesson 058).

```python
import torch.nn as nn
lstm = nn.LSTM(input_size=10, hidden_size=20, batch_first=True)
output, (h_n, c_n) = lstm(x)   # note: LSTM returns BOTH hidden and cell state
```

## GRU: a simpler, often-comparable alternative

**Gated Recurrent Unit** merges the forget and input gates into a single
**update gate**, and drops the separate cell state (folding memory directly
into the hidden state):

```
update_gate  z_t = sigmoid(W_z @ [h_(t-1), x_t])
reset_gate   r_t = sigmoid(W_r @ [h_(t-1), x_t])
candidate    h~_t = tanh(W @ [r_t * h_(t-1), x_t])
hidden state h_t = (1 - z_t) * h_(t-1) + z_t * h~_t
```

Fewer parameters than LSTM (2 gates instead of 3, no separate cell state),
often trains faster, and performs comparably on many tasks — GRU vs LSTM is
frequently an empirical choice rather than one with a clear universal
winner.

```python
gru = nn.GRU(input_size=10, hidden_size=20, batch_first=True)
output, h_n = gru(x)   # only hidden state, no separate cell state
```

## Stacking and bidirectionality (same as plain RNNs)

```python
lstm = nn.LSTM(input_size=10, hidden_size=20, num_layers=2, bidirectional=True, batch_first=True)
```

`num_layers=2` stacks LSTM layers (output of layer 1 feeds layer 2, as
inputs); `bidirectional=True` works exactly as in Lesson 045.

## Practical guidance

- **Default to LSTM or GRU over plain RNN** for any real sequence task —
  the vanishing gradient problem makes plain RNNs impractical beyond short
  sequences in most real applications.
- **GRU as a first try** if you want fewer parameters/faster training;
  **LSTM** if you have the compute budget and want to try the more
  expressive (3-gate) option — benchmark both if it matters for your task.
- Both still process sequences **strictly sequentially** (timestep `t`
  needs `h_(t-1)`) — this fundamental limitation (no parallelization across
  time during training) is untouched by gating, and is the deeper reason
  Transformers (which process all positions in parallel via attention)
  eventually displaced RNN-family models for most large-scale NLP,
  including every modern LLM.

## Where this leaves you heading into Module 8-10

You now understand the *shape* of the vanishing-gradient/long-range-
dependency problem from two angles — gating (this lesson) and skip
connections (Lesson 044). Attention (Lesson 058) solves it a third way:
by letting any position directly attend to any other position, regardless
of distance, with no sequential chain of gates or timesteps in between at
all.
