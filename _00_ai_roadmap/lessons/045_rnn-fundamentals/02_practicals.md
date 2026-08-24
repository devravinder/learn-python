# 02 — Practicals: RNN Fundamentals

1. Implement a single RNN cell's forward pass from scratch in NumPy
   (`h_t = tanh(W_xh @ x_t + W_hh @ h_(t-1) + b_h)`), and manually unroll it
   over a toy 5-timestep sequence of 3-dim vectors, printing the hidden
   state at each step.

2. Recreate the same computation using `torch.nn.RNN` and confirm the
   final hidden state matches your from-scratch version (you'll need to
   copy your NumPy weights into the PyTorch RNN's weight tensors manually —
   `rnn.weight_ih_l0.data = ...` etc. — a good exercise in understanding
   exactly what the module wraps).

3. Build a many-to-one RNN classifier: given sequences of length 10
   (`torch.randn(batch, 10, 1)`) labeled by whether their **sum** is
   positive or negative, train an RNN (`nn.RNN` -> take the final hidden
   state -> `nn.Linear` -> sigmoid) to classify them. Report accuracy.

4. **Demonstrate vanishing gradients directly**: build an RNN and a
   sequence of length 100. After one forward+backward pass on a loss
   computed only from the *final* timestep's output, print the gradient
   norm of the hidden-to-hidden weight matrix's gradient
   contribution from the *first* timestep (you'll need hooks, or simply
   compare `rnn.weight_hh_l0.grad` magnitude across different sequence
   lengths: 10, 50, 100, 200) — confirm gradient magnitude shrinks as
   sequence length grows.

5. Apply gradient clipping (`clip_grad_norm_`, max_norm=1.0) during
   training of the Q3 classifier on purposely noisy/unstable data (try a
   large learning rate like `lr=1.0` that would otherwise risk exploding
   gradients). Compare training stability (does the loss ever become `nan`)
   with and without clipping.

6. Compare a unidirectional vs bidirectional RNN (`bidirectional=True`) on
   a many-to-many tagging-style toy task: given a sequence of numbers,
   label each position `1` if it's a local maximum (greater than both
   neighbors) else `0` — a task where knowing the *next* element (not just
   past elements) is genuinely necessary. Does bidirectionality
   measurably help here?
