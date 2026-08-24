# 02 — Practicals: Alpha-Beta Pruning & Evaluation Functions

Reuse Lesson 048's tic-tac-toe `legal_moves`/`make_move`/`check_winner`.

1. Implement `alphabeta(board, player, alpha, beta)` mirroring
   `01_concepts.md`'s pseudocode, adapted to tic-tac-toe's `"X"`/`"O"`
   framing (X maximizes, O minimizes, matching Lesson 048).

2. Implement `best_move_ab(board, player)` using `alphabeta` instead of
   plain `minimax`. Run it on the empty board and confirm it picks the
   **same move** Lesson 048's plain minimax chose.

3. Count `alphabeta` calls (global counter, as in Lesson 048 Q6) for
   choosing the first move from an empty board. Compare to plain minimax's
   549,945 calls — what percentage reduction do you get?

4. Test the move-ordering claim from `01_concepts.md`: shuffle the order
   `legal_moves` returns (`random.shuffle`, fixed seed) instead of the
   natural left-to-right order, and re-count calls for the same first-move
   decision. Does call count change with different move ordering, even
   though the *chosen move* stays the same? (Try reversing the order too —
   tic-tac-toe's left-right symmetry means simple reversal may show
   *no* difference at all, which is itself worth noticing and explaining.)

5. Implement a simple material-only evaluation function for a **made-up**
   simplified board (not full chess — e.g. represent a position as a dict
   `{"white": ["pawn","pawn","knight"], "black": ["pawn","rook"]}` with
   values `{"pawn":1,"knight":3,"bishop":3,"rook":5,"queen":9}`), returning
   `sum(white values) - sum(black values)`. Test it on a few hand-made
   examples and confirm the sign matches which side has more material.

6. Install `python-chess` (`pip install python-chess`) and write
   `evaluate_material(board)` using `01_concepts.md`'s snippet on a real
   `chess.Board()`. Test it on the starting position (should be exactly
   `0` — material is symmetric) and after removing a piece manually
   (`board.remove_piece_at(...)`) to confirm the score shifts correctly.
