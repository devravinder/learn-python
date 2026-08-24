# 01 — Requirement: Chess Bot v1 (Minimax + Alpha-Beta)

## The brief

Build a chess engine that can play a full legal game against a human from
the terminal, choosing moves via depth-limited alpha-beta search over a
hand-crafted evaluation function.

## What to produce

1. **Evaluation function** (Lesson 049): combine at minimum —
   - **Material**: standard piece values (P=1, N=3, B=3, R=5, Q=9).
   - **Piece-square tables**: at least for pawns (bonus for advancing) and
     knights (bonus for centralization) — simplified tables are fine, don't
     need to hand-tune every piece perfectly.
   - **Mobility**: a small bonus per legal move available (encourages
     active piece play).

   Return a score from the side-to-move's perspective (positive = good for
   whoever's turn it is) — this is the **negamax** convention from Lesson
   048, which simplifies the search code.

2. **Search**: implement negamax with alpha-beta pruning (Lessons 048-049)
   at a configurable fixed depth (start with depth 3-4 — deeper is slower
   but stronger; benchmark the tradeoff, see Q5 below).

3. **Move ordering** (Lesson 049): before searching, sort legal moves so
   captures are tried first (`board.is_capture(move)`) — report the effect
   on nodes searched at a fixed depth, with vs without this ordering.

4. **Playable CLI**: a loop that shows the board (`print(board)` from
   `python-chess` gives a readable ASCII board for free), takes the human's
   move in UCI or SAN notation (`board.push_san(...)`), lets the bot reply,
   and detects game end (`board.is_game_over()`, `board.outcome()`).

5. **Benchmark**: play the bot against **itself** at different depths (e.g.
   depth 2 vs depth 4) for 10 games, alternating colors. Does the deeper
   search win more often? Report the win rate and note how much longer
   deeper search takes per move.

6. **Sanity checks**: verify the bot doesn't hang a queen for nothing
   against an obvious 1-move tactic (set up a simple "free queen capture"
   position with `board.set_fen(...)` and confirm the bot takes it), and
   that it can find a simple forced mate-in-2 puzzle (find one online or
   construct one) at sufficient search depth.

## Constraints

- Use `python-chess` for the rules/move generation only — the evaluation
  function and search algorithm must be your own code.
- Don't peek at `02_solutions/` before you have a playable bot yourself.
