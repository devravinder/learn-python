# 01 — Requirement: Chess Bot v3 (Self-Play + MCTS)

## The brief

Implement the full AlphaZero-style loop from Lesson 054 for chess:
a combined policy+value network, MCTS guided by that network (PUCT
selection, network value instead of random rollouts), self-play game
generation, and training the network on the self-play data — repeated for
several generations.

## What to produce

1. **Combined policy+value network**: extend Project 009's `(12,8,8)` board
   encoding + CNN trunk with **two output heads**:
   - **Value head**: same as Project 009 — a single scalar in
     roughly `[-1, 1]`.
   - **Policy head**: a probability distribution over a *fixed-size* move
     encoding. Chess has no fixed small action space like tic-tac-toe's 9
     cells — use a standard simplification: encode moves as
     `(from_square, to_square)` pairs (`64*64 = 4096` possible, ignoring
     underpromotion choice for simplicity — promote to queen always) and
     mask out illegal moves at each position before normalizing with
     softmax.

2. **Network-guided MCTS (PUCT)**: adapt Lesson 053's MCTS —
   - Selection uses **PUCT** instead of plain UCB1:
     `PUCT(s,a) = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))`
     (`P(s,a)` is the network's prior probability for action `a`).
   - No random rollout — when a node is expanded, run the network once,
     use its value output directly, and use its policy output to set prior
     probabilities `P(s,a)` for the new node's children.

3. **Self-play data generation**: play games using MCTS (guided by the
   current network) for both sides. For each position visited, record
   `(board_tensor, mcts_visit_distribution, game_outcome)` — the visit
   distribution (normalized) is the **policy target**, and the eventual
   outcome (+1/-1/0 from that position's side-to-move perspective) is the
   **value target**.

4. **Training**: combined loss (Lesson 054): value MSE + policy
   cross-entropy against the MCTS visit distribution, plus light L2
   weight decay. Train for a modest number of epochs per generation.

5. **Iterate for at least 3 generations**: generation 0 = randomly
   initialized network; after each generation's self-play + training,
   save a checkpoint. Use a small self-play budget per generation to keep
   this tractable (e.g. 20-50 games, 50-200 MCTS simulations per move —
   tune based on your actual hardware and patience).

6. **Measure real improvement (the actual deliverable)**: play generation
   N's network against generation 0 (random-init) head-to-head, both using
   the same MCTS search settings, for at least 20 games. Report the win
   rate. **This is the number that matters for this project** — a rising
   win rate across generations is the concrete, measurable proof the
   self-play loop is working as intended.

7. **Compare to Project 008/009**: play your best generation's bot (network
   value + MCTS, no hand-crafted eval at all) against Project 008's v1
   at a modest depth. Given the scope note above, do **not** expect to win
   — report the actual result honestly, and reflect on what
   compute/training scale would plausibly be needed to close the gap
   (AlphaZero's paper reports its own training compute — look it up and
   compare orders of magnitude to what you actually ran).

## Constraints

- The full AlphaZero algorithm's mechanism must be genuinely present
  (network-guided MCTS, self-play data collection, combined-loss training,
  generational iteration) — a fixed/small scale is expected and fine, but
  don't skip a core mechanism (e.g. don't fall back to random rollouts "to
  save time," since that's Lesson 053's algorithm, not this one).
- Don't peek at `02_solutions/` before you have your own working self-play
  loop and generation-over-generation win-rate measurement.
