# 03 — Solutions: Game Trees & Minimax

*(This code was actually run to produce the numbers below.)*

## 1. Board representation and helpers

```python
def legal_moves(board):
    return [i for i, c in enumerate(board) if c == " "]

def make_move(board, idx, player):
    new_board = board[:]
    new_board[idx] = player
    return new_board

WIN_LINES = [
    (0,1,2),(3,4,5),(6,7,8),
    (0,3,6),(1,4,7),(2,5,8),
    (0,4,8),(2,4,6),
]

def check_winner(board):
    for a, b, c in WIN_LINES:
        if board[a] != " " and board[a] == board[b] == board[c]:
            return board[a]
    if " " not in board:
        return "draw"
    return None
```

## 2–3. Minimax and best_move

```python
CALL_COUNT = 0

def minimax(board, player):
    global CALL_COUNT
    CALL_COUNT += 1
    winner = check_winner(board)
    if winner == "X": return 1
    if winner == "O": return -1
    if winner == "draw": return 0

    if player == "X":
        best = float("-inf")
        for m in legal_moves(board):
            best = max(best, minimax(make_move(board, m, "X"), "O"))
        return best
    else:
        best = float("inf")
        for m in legal_moves(board):
            best = min(best, minimax(make_move(board, m, "O"), "X"))
        return best

def best_move(board, player):
    scored = []
    for m in legal_moves(board):
        val = minimax(make_move(board, m, player), "O" if player == "X" else "X")
        scored.append((val, m))
    return max(scored)[1] if player == "X" else min(scored)[1]
```

## 4. Minimax vs itself

```python
board = [" "] * 9
player = "X"
while check_winner(board) is None:
    m = best_move(board, player)
    board = make_move(board, m, player)
    player = "O" if player == "X" else "X"
print(check_winner(board))
```

**Actual output: `draw`** — confirming the well-known result that perfect
tic-tac-toe play from both sides always ends in a draw, exactly as Lesson
048's "minimax finds the game-theoretically optimal result" claim predicts.

```text
['O', 'X', 'X']
['X', 'O', 'O']
['O', 'X', 'X']
```

## 5. Minimax vs random opponent (100 games)

```python
import random

def random_move(board):
    return random.choice(legal_moves(board))

random.seed(0)
results = {"minimax_win": 0, "draw": 0, "minimax_loss": 0}
for game in range(100):
    board = [" "] * 9
    minimax_is_x = (game % 2 == 0)
    player = "X"
    while check_winner(board) is None:
        m = best_move(board, player) if (player == "X") == minimax_is_x else random_move(board)
        board = make_move(board, m, player)
        player = "O" if player == "X" else "X"
    winner = check_winner(board)
    if winner == "draw":
        results["draw"] += 1
    elif (winner == "X") == minimax_is_x:
        results["minimax_win"] += 1
    else:
        results["minimax_loss"] += 1

print(results)
```

**Actual output: `{'minimax_win': 96, 'draw': 4, 'minimax_loss': 0}`** —
**zero losses across 100 games**, exactly what minimax guarantees against
any opponent (a suboptimal random player will lose or draw, never win,
against optimal play).

## 6. Node count for the first move

```python
CALL_COUNT = 0
board = [" "] * 9
best_move(board, "X")
print(CALL_COUNT)
```

**Actual output: `549945`** — over half a million `minimax` calls just to
choose the *first* move of a game with only 9 cells. This is the exact,
measured cost that Lesson 049's alpha-beta pruning will cut dramatically
(often by 90%+) — same chosen move, far less computation — and it's why
plain minimax alone is completely infeasible for chess's vastly larger tree
without both pruning and a much shallower depth limit.
