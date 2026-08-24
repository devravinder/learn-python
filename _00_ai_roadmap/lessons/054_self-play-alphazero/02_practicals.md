# 02 — Practicals: Self-Play & AlphaZero-Style Training

A deliberately small-scale, fully-runnable version of the self-play loop on
tic-tac-toe — using Lesson 038's micrograd-style `Value`/`MLP` engine
(no PyTorch needed), a **value network only** (skip the policy head, to
keep this tractable as a lesson exercise — Project 010 implements the full
policy+value version at chess scale).

1. Reuse Lesson 048's tic-tac-toe logic. Write `encode(board, player)`:
   convert a board into 9 numbers, `+1` for the current player's pieces,
   `-1` for the opponent's, `0` for empty — a simple, fixed-size numeric
   representation a network can consume.

2. Generate **Generation 0 self-play data**: play `N` complete games of
   **uniformly random** moves (no network involved yet — this is the
   "before any training" starting point). For every position in every
   game, record `(encode(board, player_to_move), z)` where `z` is `+1` if
   `player_to_move` eventually won, `-1` if they lost, `0` for a draw.

3. Build a small value network: `MLP(9, [8, 1])` (Lesson 038's classes,
   `tanh` hidden layer, linear output so it can predict any value in
   roughly `[-1, 1]`). Train it with MSE loss (Lesson 020) against the
   generation-0 targets from Q2, using minibatch gradient descent
   (Lesson 015).

4. **Sanity check before trusting it**: evaluate the trained network on a
   hand-picked "obviously good for X" position (X has two in a row, e.g.
   `["X","X"," "," "," "," "," "," "," "]`) and an "obviously bad for X"
   position (`["O","O"," "," "," "," "," "," "," "]`). Does the network
   correctly value the first higher than the second? If not — **don't
   assume the code is broken yet**; try more training games and more
   epochs first (Q5) before debugging.

5. Retrain with substantially more self-play games (e.g. 300 instead of a
   smaller first attempt) and more training epochs. Does the sanity check
   from Q4 pass now? Compare training-set MSE loss before/after both
   attempts.

6. Explain, in 3-4 sentences, why a value network trained **only on
   outcomes of fully random games** is a weaker training signal than
   AlphaZero's actual approach (training against **MCTS-improved** move
   distributions, Lesson 053 + `01_concepts.md`) — and why you'd expect
   Project 010's iterative self-play (network improves -> better self-play
   games -> better training data -> better network -> repeat) to eventually
   produce much stronger play than this single-pass, random-data version.
