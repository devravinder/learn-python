# Reference Solutions

```bash
python tictactoe_bot.py             # play interactively (you're O, bot is X)
python tictactoe_bot.py --analyze   # Q3-Q4 analysis
```

*(Output below was actually produced by running this exact script.)*

## Q3: depth-limited vs full-depth (50 games each)

```text
depth=1 vs full-depth: {'depth_limited_win': 0, 'draw': 0, 'depth_limited_loss': 50}
depth=2 vs full-depth: {'depth_limited_win': 0, 'draw': 50, 'depth_limited_loss': 0}
```

**Depth 1 loses every single game** — a 1-ply lookahead with this simple
open-lines heuristic is too myopic to avoid traps a perfect player sets up.
**Depth 2 already draws every game** — matching full-depth (perfect) play
exactly, despite searching only 2 moves ahead with a heuristic instead of
solving the game fully. This is a genuinely encouraging, realistic result:
a decent heuristic plus a shallow search can already match optimal play on
a small game — exactly the bet Project 008's Chess Bot v1 makes at a much
larger scale (a heuristic evaluation + a search depth chess engines can
actually afford).

## Q4: node count vs depth

```text
depth=1: 9 calls
depth=2: 81 calls
depth=3: 272 calls
depth=4: 1136 calls
depth=5: 3801 calls
depth=9: 34202 calls
```

Growth is clearly super-linear (roughly exponential, though pruning keeps
it well below the theoretical worst case) — going from depth 4 to depth 5
alone more than triples the call count. This is precisely why chess
engines can only search a handful of moves ahead within a practical time
budget: each additional ply of depth costs several times more computation
than the last, chess's branching factor (~35 legal moves per position,
vs. tic-tac-toe's ≤9) makes this growth far steeper still.

## Q6: what carries over to Project 008

The **search algorithm** (`alphabeta`, including depth limiting and the
`alpha >= beta` cutoff) carries over almost unchanged — it doesn't know or
care what game it's searching, only that positions have legal moves and an
evaluation. The **heuristic function** (`heuristic()`, open winning lines)
is entirely tic-tac-toe-specific and must be completely redesigned for
chess — material count, piece-square tables, king safety, etc. (Lesson
049) — since "open winning lines" has no chess equivalent. This split
(generic search vs. game-specific evaluation) is exactly how Project 008 is
structured, and exactly why Project 009 can later swap in a *learned*
evaluation function without touching the search code at all.
