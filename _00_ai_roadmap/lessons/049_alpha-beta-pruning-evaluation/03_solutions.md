# 03 — Solutions: Alpha-Beta Pruning & Evaluation Functions

*(This code was actually run to produce the numbers below.)*

## 1–2. Alpha-beta for tic-tac-toe

```python
AB_CALLS = 0

def alphabeta(board, player, alpha, beta):
    global AB_CALLS
    AB_CALLS += 1
    winner = check_winner(board)
    if winner == "X": return 1
    if winner == "O": return -1
    if winner == "draw": return 0

    moves = legal_moves(board)
    if player == "X":
        value = float("-inf")
        for m in moves:
            value = max(value, alphabeta(make_move(board, m, "X"), "O", alpha, beta))
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value
    else:
        value = float("inf")
        for m in moves:
            value = min(value, alphabeta(make_move(board, m, "O"), "X", alpha, beta))
            beta = min(beta, value)
            if alpha >= beta:
                break
        return value

def best_move_ab(board, player):
    scored = []
    for m in legal_moves(board):
        val = alphabeta(make_move(board, m, player), "O" if player == "X" else "X", float("-inf"), float("inf"))
        scored.append((val, m))
    return max(scored)[1] if player == "X" else min(scored)[1]

board = [" "] * 9
AB_CALLS = 0
m = best_move_ab(board, "X")
print(m, AB_CALLS)
```

**Actual output: move `8`, `30709` calls.** Lesson 048's plain minimax also
chose move `8` for the first move — **identical move, from an algorithm
that's provably guaranteed to never change the answer.**

## 3. Reduction vs plain minimax

```
reduction = (549945 - 30709) / 549945 * 100
```

**Actual result: a 94.4% reduction** in calls (from 549,945 down to
30,709) for the exact same decision — a striking, concrete demonstration
of why alpha-beta pruning is standard in every real game-search engine,
including Project 008's Chess Bot.

## 4. Move ordering effects

```python
import random

def legal_moves_reversed(board):
    return list(reversed(legal_moves(board)))

# ... alphabeta_rev using legal_moves_reversed ...

AB_CALLS = 0
best_move_ab_rev(board, "X")   # same alphabeta logic, reversed move order
print(AB_CALLS)   # 30709 -- IDENTICAL to forward order

random.seed(1)
def legal_moves_shuffled(board):
    moves = legal_moves(board)
    random.shuffle(moves)
    return moves

# ... alphabeta_shuf using legal_moves_shuffled ...
AB_CALLS = 0
best_move_ab_shuf(board, "X")
print(AB_CALLS)   # 28486 -- DIFFERENT from both forward and reversed
```

**Actual results**: reversed order gave **exactly 30,709 calls — identical**
to forward order, while a random shuffle gave **28,486 calls — different
from both**. The reversal result isn't a bug: tic-tac-toe's board has a
left-right mirror symmetry, so reversing the move order from an empty board
explores a mirror-image tree of exactly the same shape and size. The random
shuffle breaks that symmetry and lands on a genuinely different call count
— confirming `01_concepts.md`'s claim that move order affects pruning
efficiency, while also surfacing a real, honest caveat: *symmetric*
reorderings of a *symmetric* position won't demonstrate the effect, so
don't be misled if your first attempt (like the simple reversal here) shows
no difference.

## 5. Toy material evaluation

```python
VALUES = {"pawn": 1, "knight": 3, "bishop": 3, "rook": 5, "queen": 9}

def evaluate_toy(position):
    white = sum(VALUES[p] for p in position["white"])
    black = sum(VALUES[p] for p in position["black"])
    return white - black

print(evaluate_toy({"white": ["pawn", "pawn", "knight"], "black": ["pawn", "rook"]}))
# 1+1+3 - (1+5) = 5 - 6 = -1  (black has a slight material edge here)
```

## 6. Real chess material evaluation with `python-chess`

```python
import chess

PIECE_VALUES = {"P": 1, "N": 3, "B": 3, "R": 5, "Q": 9, "K": 0}

def evaluate_material(board):
    score = 0
    for square in board.piece_map().values():
        value = PIECE_VALUES[square.symbol().upper()]
        score += value if square.color else -value
    return score

board = chess.Board()
print(evaluate_material(board))   # 0 -- starting position is exactly symmetric

board.remove_piece_at(chess.E2)   # remove a white pawn
print(evaluate_material(board))   # -1 -- White is now down a pawn, score reflects it
```

Removing a White pawn shifts the score from `0` to `-1` (using the
convention `White=True` contributes positively) — confirming the
evaluation function correctly tracks material balance, exactly the
function Project 008's Chess Bot v1 will call at every leaf of its
alpha-beta search.
