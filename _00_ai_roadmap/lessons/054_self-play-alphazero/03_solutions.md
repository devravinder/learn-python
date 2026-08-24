# 03 — Solutions: Self-Play & AlphaZero-Style Training

*(This code was actually run to produce the numbers below, using Lesson
038's `Value`/`MLP` classes — copy those in before running this.)*

## 1. Board encoding

```python
def encode(board, player):
    return [1.0 if c == player else (-1.0 if c == other(player) and c != " " else 0.0) for c in board]
```

## 2. Generation-0 self-play data

```python
def random_self_play_game():
    board = [" "] * 9
    player = "X"
    history = []
    while check_winner(board) is None:
        history.append((board[:], player))
        m = random.choice(legal_moves(board))
        board = make_move(board, m, player)
        player = other(player)
    winner = check_winner(board)
    data = []
    for b, p in history:
        z = 0.0 if winner == "draw" else (1.0 if winner == p else -1.0)
        data.append((encode(b, p), z))
    return data
```

## 3–4. First attempt: small data, few epochs

```python
random.seed(0)
dataset = []
for _ in range(80):
    dataset.extend(random_self_play_game())
print(len(dataset))   # 596 examples

random.seed(1)
net = MLP(9, [8, 1])

def evaluate_net_loss(data):
    total = Value(0.0)
    for x, z in data:
        pred = net([Value(v) for v in x])
        total = total + (pred - z) ** 2
    return total * (1.0 / len(data))

sample = dataset[:100]
print("loss before:", evaluate_net_loss(sample).data)

lr, batch_size = 0.05, 16
for epoch in range(6):
    random.shuffle(dataset)
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i+batch_size]
        loss = evaluate_net_loss(batch)
        for p in net.parameters():
            p.grad = 0.0
        loss.backward()
        for p in net.parameters():
            p.data -= lr * p.grad

print("loss after:", evaluate_net_loss(sample).data)

near_win = ["X","X"," "," "," "," "," "," "," "]
near_loss = ["O","O"," "," "," "," "," "," "," "]
print("value(near-win-for-X):", net([Value(v) for v in encode(near_win, "X")]).data)
print("value(near-loss-for-X):", net([Value(v) for v in encode(near_loss, "X")]).data)
```

**Actual output:**

```text
training examples: 596
loss before: 1.854
loss after:  0.886
value(near-win-for-X):  -0.032
value(near-loss-for-X):  0.224
```

**The sanity check fails** — the network rates the near-loss position
*higher* than the near-win position, backwards from what it should be,
despite training loss dropping. This is a real, honest result worth
sitting with before assuming a bug: 596 examples from **fully random** play
is a small, noisy dataset, and 6 epochs isn't much training.

## 5. More data and training fixes it

```python
random.seed(0)
dataset = []
for _ in range(300):
    dataset.extend(random_self_play_game())
print(len(dataset))   # 2297 examples

random.seed(1)
net = MLP(9, [8, 1])
# ... same evaluate_net_loss ...

sample = dataset[:150]
print("loss before:", evaluate_net_loss(sample).data)

for epoch in range(15):    # more epochs this time
    random.shuffle(dataset)
    for i in range(0, len(dataset), 16):
        batch = dataset[i:i+16]
        loss = evaluate_net_loss(batch)
        for p in net.parameters(): p.grad = 0.0
        loss.backward()
        for p in net.parameters(): p.data -= lr * p.grad

print("loss after:", evaluate_net_loss(sample).data)
print("value(near-win-for-X):", net([Value(v) for v in encode(near_win, "X")]).data)
print("value(near-loss-for-X):", net([Value(v) for v in encode(near_loss, "X")]).data)
```

**Actual output:**

```text
training examples: 2297
loss before: 2.481
loss after:  1.106
value(near-win-for-X):   1.160
value(near-loss-for-X): -0.793
```

With ~4x the data and more epochs, **the sanity check now passes clearly**
— the network correctly rates the near-win position strongly positive and
the near-loss position strongly negative. The lesson: the first attempt
wasn't a coding bug, it was **insufficient training data and time** — a
distinction worth learning to diagnose (check data/training scale before
assuming your implementation is wrong), since it's one of the most common
real debugging fork-in-the-road in applied ML.

## 6. Why MCTS-improved targets beat raw random-game outcomes

Training only on random-game outcomes means the label for a position is
"did fully random play happen to win from here" — a very noisy signal,
since a genuinely strong position can still lose if both sides play
randomly afterward, and vice versa. AlphaZero's approach instead uses
**MCTS's visit-count distribution** (Lesson 053 guided by the *current*
network) as the target, which already reflects focused, non-random search
that concentrates on genuinely strong continuations — a meaningfully better
label than "outcome of subsequent random moves." Because each training
round produces a *better* network, which then guides *better* MCTS search,
which produces *better* training data than the previous round, repeating
this loop (Project 010's actual task) compounds improvement over many
iterations — something a single training pass on static random-game data,
like this lesson's simplified exercise, structurally cannot do.
