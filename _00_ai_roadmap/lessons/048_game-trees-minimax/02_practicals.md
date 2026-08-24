# 02 — Practicals: Game Trees & Minimax

Build a complete tic-tac-toe minimax player — small enough to search the
**entire** game tree (no depth limit needed), so you can verify it plays
provably optimally.

1. Represent a board as a list of 9 cells (`" "`, `"X"`, `"O"`). Write
   `legal_moves(board)` (empty cell indices), `make_move(board, idx, player)`
   (returns a new board), `check_winner(board)` (returns `"X"`, `"O"`,
   `"draw"`, or `None`).

2. Implement `minimax(board, player)` (no depth limit — tic-tac-toe's tree
   is small enough to fully explore) that returns `+1` if `"X"` wins,
   `-1` if `"O"` wins, `0` for a draw, recursing to terminal positions.

3. Implement `best_move(board, player)`: try every legal move, run
   `minimax` on the resulting position, and return the move with the best
   outcome for `player` (`X` maximizes, `O` minimizes — matching Lesson
   048's MAX/MIN framing).

4. Play the minimax player against **itself** from an empty board. Confirm
   the result is always a **draw** — the well-known result that perfect
   tic-tac-toe play from both sides never loses, matching real-world
   experience playing the game well.

5. Play the minimax player against a **random-move player** (picks a
   uniformly random legal move) 100 times, with the minimax player going
   first half the time and second half the time. Confirm the minimax player
   never loses (wins or draws every game).

6. Count how many `minimax` calls (add a global counter) are needed to
   choose the very first move from an empty board. This is the number
   Lesson 049's alpha-beta pruning will dramatically reduce without
   changing the chosen move at all.
