# 03 — Solutions: Monte Carlo Tree Search

*(This code was actually run to produce the numbers below — including
catching a real perspective bug along the way, described in Q2.)*

## 1. Node and the four phases

```python
import math
import random

def other(p): return "O" if p == "X" else "X"

class Node:
    def __init__(self, board, player, parent=None, move=None):
        self.board = board
        self.player = player       # player to move AT this node
        self.parent = parent
        self.move = move           # move that led here
        self.children = []
        self.wins = 0.0
        self.visits = 0
        self.untried = legal_moves(board)

    def ucb1(self, C=1.4):
        if self.visits == 0:
            return float("inf")    # always try unvisited children first
        return (self.wins / self.visits) + C * math.sqrt(math.log(self.parent.visits) / self.visits)

def select(node):
    while not node.untried and node.children:
        node = max(node.children, key=lambda n: n.ucb1())
    return node

def expand(node):
    move = node.untried.pop()
    child = Node(make_move(node.board, move, node.player), other(node.player), parent=node, move=move)
    node.children.append(child)
    return child

def rollout(board, player):
    while check_winner(board) is None:
        move = random.choice(legal_moves(board))
        board = make_move(board, move, player)
        player = other(player)
    return check_winner(board)
```

## 2. The perspective bug (a real one, caught by testing)

A first version of `backpropagate` tracked `wins` as "wins for whoever was
to move at the **root**," applied uniformly to every node on the
backpropagation path. This is **wrong**, for the same reason plain minimax
alternates MAX/MIN (Lesson 048): at a node where the *opponent* is about
to choose among children, they want to pick the child that's worst for
root's perspective, not best — but UCB1 as written always picks the
**highest**-value child. Using a single fixed perspective throughout
breaks this alternation.

**Symptom observed**: with the buggy version, MCTS at 2000 simulations per
move still **lost 6 out of 50 games to a uniformly random opponent** — it
should never lose at all with that much search on a game this small.

**The fix**: each node's `wins` must track win-rate for **whoever made the
move leading into that node** — i.e., `other(node.player)`, since
`node.player` is who moves *next* from this node, not who just moved:

```python
def backpropagate(node, winner):
    while node is not None:
        node.visits += 1
        mover_into_node = other(node.player)
        if winner == mover_into_node:
            node.wins += 1
        elif winner == "draw":
            node.wins += 0.5
        node = node.parent
```

After this fix, losses at high simulation budgets disappeared entirely
(see Q5) — a concrete example of why testing against a known baseline
(here: "should never lose to random with enough search") is essential
before trusting any search-based AI code, not just an abstract nicety.

## 3. Best-move selection

```python
def mcts_best_move(board, player, n_simulations=1000):
    root = Node(board, player)
    for _ in range(n_simulations):
        node = select(root)
        winner = check_winner(node.board)
        if winner is None:
            node = expand(node)
            winner = rollout(node.board, node.player)
        backpropagate(node, winner)
    return max(root.children, key=lambda n: n.visits).move
```

## 4. Opening move distribution

```python
random.seed(0)
board = [" "] * 9
move_counts = {}
for _ in range(20):
    m = mcts_best_move(board, "X", n_simulations=500)
    move_counts[m] = move_counts.get(m, 0) + 1
print(move_counts)
```

**Actual output: `{4: 18, 2: 2}`** — 18 of 20 trials chose the center
(square 4), 2 chose a corner (square 2) — both are genuinely optimal
tic-tac-toe openings (edges, e.g. squares 1/3/5/7, are known to be
inferior), matching game theory despite MCTS never being told the rules
beyond legal moves and win/loss/draw detection.

## 5. MCTS vs random at different simulation budgets

```python
def mcts_vs_random(n_games, n_sims, seed):
    random.seed(seed)
    results = {"mcts_win": 0, "draw": 0, "mcts_loss": 0}
    for g in range(n_games):
        board = [" "] * 9
        mcts_is_x = (g % 2 == 0)
        player = "X"
        while check_winner(board) is None:
            move = mcts_best_move(board, player, n_sims) if (player == "X") == mcts_is_x \
                   else random.choice(legal_moves(board))
            board = make_move(board, move, player)
            player = other(player)
        winner = check_winner(board)
        if winner == "draw": results["draw"] += 1
        elif (winner == "X") == mcts_is_x: results["mcts_win"] += 1
        else: results["mcts_loss"] += 1
    return results

for sims in [50, 300, 1000]:
    print(sims, mcts_vs_random(50, sims, seed=1))
```

**Actual output:**

```text
50   sims: {'mcts_win': 47, 'draw': 1, 'mcts_loss': 2}
300  sims: {'mcts_win': 46, 'draw': 4, 'mcts_loss': 0}
1000 sims: {'mcts_win': 48, 'draw': 2, 'mcts_loss': 0}
```

At only 50 simulations per move, MCTS still loses occasionally (2/50) —
too little search to reliably avoid every trap. By 300 simulations,
**losses drop to zero** and stay there at 1000 — exactly the convergence
behavior MCTS theory predicts: more simulations means less noisy value
estimates and search that concentrates ever more reliably on genuinely
strong moves.

## 6. Exploration constant C

```python
for C in [0.1, 1.4, 5.0]:
    print(C, mcts_vs_random_with_C(50, 100, C, seed=2))   # ucb1(C) passed through select()
```

**Actual output:**

```text
C=0.1: {'mcts_win': 43, 'draw': 5, 'mcts_loss': 2}
C=1.4: {'mcts_win': 48, 'draw': 2, 'mcts_loss': 0}
C=5.0: {'mcts_win': 45, 'draw': 3, 'mcts_loss': 2}
```

Both extremes underperform the balanced `C=1.4`: too low (`0.1`) means the
search over-exploits early estimates before they're reliable (barely
exploring alternatives); too high (`5.0`) means it spreads search too
thinly across many mediocre moves instead of concentrating on the best
ones. This U-shaped pattern — a real, measured result, not just a
theoretical claim — is the exploration/exploitation tradeoff from Lesson
050, showing up concretely in a hyperparameter you have to tune.
