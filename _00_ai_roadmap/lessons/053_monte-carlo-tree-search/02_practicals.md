# 02 — Practicals: Monte Carlo Tree Search

Reuse Lesson 048's tic-tac-toe `legal_moves`/`make_move`/`check_winner`.

1. Implement a `Node` class (board, player-to-move, parent, move, children,
   wins, visits, untried moves) and the four MCTS phases from
   `01_concepts.md`: `select` (UCB1), `expand`, `rollout` (uniform random
   playout), `backpropagate`.

2. **Careful perspective bookkeeping**: when backpropagating a result,
   each node's `wins` should track win-rate for **whoever made the move
   that led to that node** (not a single fixed "root player" throughout) —
   otherwise selection picks moves that are good for the wrong side at
   alternating tree levels, similar to Lesson 048's MAX/MIN alternation.
   Get this right before trusting any results below.

3. Implement `mcts_best_move(board, player, n_simulations)`: run the four
   phases `n_simulations` times from the given board, then return the
   **most-visited** child's move (not the highest win-rate child — visit
   count is the standard, more robust choice, since a move visited only
   once with a lucky win looks artificially good by win-rate alone).

4. From an empty board, run `mcts_best_move` 20 times with 500 simulations
   each and tally which move gets chosen. Tic-tac-toe's known-optimal
   opening moves are the center and the 4 corners — does MCTS's move
   distribution land only on those squares?

5. Play MCTS against a random-move player, 50 games (alternating who goes
   first), at three simulation budgets: 50, 300, and 1000 simulations per
   move. Report win/draw/loss counts for each budget. Does MCTS ever lose
   at the lowest budget? Does that stop happening as the budget increases?

6. Increase the UCB1 exploration constant `C` from 1.4 to a much larger
   value (e.g. 5) and a much smaller value (e.g. 0.1). At a fixed, modest
   simulation budget (e.g. 100), does either extreme visibly hurt play
   quality against the random opponent, compared to `C=1.4`? Relate your
   observation to the exploration/exploitation tradeoff from Lesson 050.
