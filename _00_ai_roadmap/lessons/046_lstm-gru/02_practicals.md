# 02 — Practicals: LSTM & GRU

1. Implement a single LSTM cell's forward pass from scratch in NumPy
   (all 4 equations from `01_concepts.md`: forget/input/output gates,
   candidate, cell state, hidden state). Unroll it over a 5-timestep toy
   sequence and print `h_t` and `C_t` at each step.

2. Design a synthetic "long-range dependency" task: sequences of length 50
   where the label depends **only on the first element** (e.g. label = 1 if
   `sequence[0] > 0` else 0, with everything else random noise). Train a
   plain `nn.RNN`, an `nn.LSTM`, and an `nn.GRU` (all many-to-one, same
   hidden size) on this task. Compare final accuracy — does the plain RNN
   struggle more than LSTM/GRU to remember information from 49 steps back?

3. Repeat Q2 with sequence length 200 instead of 50. Does the gap between
   plain RNN and LSTM/GRU widen, shrink, or stay similar?

4. Compare parameter counts of `nn.RNN(10, 20)`, `nn.GRU(10, 20)`, and
   `nn.LSTM(10, 20)` (`sum(p.numel() for p in module.parameters())`).
   Confirm LSTM has roughly 4x an RNN's parameters (4 sets of gate weights)
   and GRU has roughly 3x (3 sets) for the same input/hidden size.

5. Train an LSTM-based character-level text generator (a **preview** of
   Lesson 063a's `makemore`-style modeling) on a short text sample (a
   paragraph or two of any text you like): given the previous character,
   predict the next one. After training, generate 100 characters by
   repeatedly sampling from the model's output distribution, starting from
   a random seed character. Does the output look vaguely word-like, even
   if nonsensical?

6. Compare training wall-clock time for a `num_layers=1` vs `num_layers=3`
   stacked LSTM on the same data. Does depth meaningfully slow down
   training here, and does it improve the long-range task's accuracy from
   Q2/Q3?
