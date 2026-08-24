# 01 — Questions

1. Write a playable command-line tic-tac-toe game: human vs. the
   alpha-beta bot from Lesson 049 (full-depth, since tic-tac-toe's tree is
   small). Print the board after every move, read the human's move from
   `input()`, validate it's legal, and announce the winner/draw at the end.

2. Add a **depth limit** parameter to `alphabeta` (per Lesson 048/049's
   pattern for larger games): stop recursing once `depth == 0`, and at that
   point return a **heuristic evaluation** instead of continuing to a true
   terminal state. Design a simple tic-tac-toe heuristic: count
   `(X's potential winning lines still open) - (O's potential winning lines
   still open)`.

3. Compare the depth-limited bot (try `depth=1` and `depth=2`) against the
   full-depth bot from Lesson 049 in 50 games each (depth-limited bot
   playing both X and O across games, vs. the full-depth bot). Does the
   depth-limited bot ever lose? At what depth does it start playing
   provably-perfect (i.e. never losing, matching Lesson 048 Q4/Q5's
   result)?

4. Count `alphabeta` calls for the first move at `depth=1`, `depth=2`, and
   full depth (9). Plot calls vs depth (log scale on the y-axis). Confirm
   call count grows roughly exponentially with depth — the exact reason
   real chess engines can only search a handful of moves ahead within a
   reasonable time budget.

5. Modify the heuristic from Q2 to also give a bonus for controlling the
   center square and corners (common real tic-tac-toe strategy). Does this
   improved heuristic let a *shallower* depth limit (e.g. depth=1) perform
   better than the simpler heuristic did at the same depth?

6. Write 3-4 sentences connecting what you just built to Project 008: which
   parts of this assignment's code (the search algorithm vs. the evaluation
   function) will carry over almost unchanged to a real chess bot, and
   which parts (the heuristic itself) will need to be completely
   redesigned for chess's much richer position structure?
