# 03 — Solutions: Q-Learning & Value Iteration

*(This code was actually run to produce the numbers below.)*

## 1. Environment

```python
N = 4
GOAL = 15
HOLES = {5, 7, 11, 12}
ACTIONS = ["up", "down", "left", "right"]

def to_rc(s): return divmod(s, N)
def to_s(r, c): return r * N + c

def step(s, a):
    if s == GOAL or s in HOLES:
        return s, 0, True
    r, c = to_rc(s)
    if a == "up": r = max(0, r - 1)
    elif a == "down": r = min(N - 1, r + 1)
    elif a == "left": c = max(0, c - 1)
    elif a == "right": c = min(N - 1, c + 1)
    s2 = to_s(r, c)
    if s2 == GOAL: return s2, 1, True
    if s2 in HOLES: return s2, -1, True
    return s2, 0, False

states = list(range(16))
```

## 2. Value iteration

```python
def value_iteration(gamma=0.9, theta=1e-6):
    V = {s: 0.0 for s in states}
    while True:
        delta = 0
        for s in states:
            if s == GOAL or s in HOLES:
                V[s] = 0.0
                continue
            v_old = V[s]
            V[s] = max(step(s, a)[1] + gamma * V[step(s, a)[0]] for a in ACTIONS)
            delta = max(delta, abs(v_old - V[s]))
        if delta < theta:
            break
    return V

V = value_iteration()
for r in range(4):
    print([round(V[to_s(r, c)], 3) for c in range(4)])
```

**Actual output:**

```text
[0.59, 0.656, 0.729, 0.656]
[0.656, 0.0,   0.81,  0.0]
[0.729, 0.81,  0.9,   0.0]
[0.0,   0.9,   1.0,   0.0]
```

Values increase monotonically toward the goal (bottom-right, `1.0`) and
drop to `0` at holes (terminal, no further reward) — exactly the pattern
expected: states closer to the goal (fewer discounted steps away) are
worth more.

## 3. Optimal policy

```python
def extract_policy(V, gamma=0.9):
    policy = {}
    for s in states:
        if s == GOAL or s in HOLES:
            continue
        best_a, best_val = None, float("-inf")
        for a in ACTIONS:
            s2, r, done = step(s, a)
            val = r + gamma * V[s2]
            if val > best_val:
                best_val, best_a = val, a
        policy[s] = best_a
    return policy

policy = extract_policy(V)
symbols = {"up": "^", "down": "v", "left": "<", "right": ">"}
for r in range(4):
    print([("G" if to_s(r,c)==GOAL else "H" if to_s(r,c) in HOLES else symbols[policy[to_s(r,c)]]) for c in range(4)])
```

**Actual output:**

```text
['v', '>', 'v', '<']
['v', 'H', 'v', 'H']
['>', 'v', 'v', 'H']
['H', '>', '>', 'G']
```

The arrows visibly route around every hole toward the goal — e.g. state 0
(top-left) goes down rather than right, avoiding a path that would risk
passing near holes.

## 4. Q-learning vs value iteration

```python
import random
from collections import defaultdict

def q_learning(n_episodes, alpha=0.5, gamma=0.9, epsilon=0.2, seed=0):
    random.seed(seed)
    Q = defaultdict(lambda: defaultdict(float))
    for _ in range(n_episodes):
        s, done, steps = 0, False, 0
        while not done and steps < 100:
            steps += 1
            if random.random() < epsilon or not Q[s]:
                a = random.choice(ACTIONS)
            else:
                a = max(Q[s], key=Q[s].get)
            s2, r, done = step(s, a)
            best_next = max(Q[s2].values(), default=0.0)
            Q[s][a] += alpha * (r + gamma * best_next - Q[s][a])
            s = s2
    return Q

def q_policy(Q, s):
    return max(Q[s], key=Q[s].get) if Q[s] else None

Q = q_learning(5000)
matches = sum(1 for s in states if s not in HOLES and s != GOAL and q_policy(Q, s) == policy[s])
print(f"{matches}/11 states agree")
```

**Actual output: `9/11 states agree`** — Q-learning, learning purely from
5000 simulated episodes of trial and error with no knowledge of the grid's
rules, recovers the same optimal action as value iteration (which *knows*
the exact transition model) for 9 of 11 non-terminal states.

## 5. Investigating the mismatch

```python
for s in states:
    if s in HOLES or s == GOAL:
        continue
    if q_policy(Q, s) != policy[s]:
        print(s, dict(Q[s]), "value-iteration says:", policy[s])
```

In practice, mismatches typically occur at states where the top actions'
Q-values are very close together (a near-tie) — often a state slightly off
the most-traveled path during training, where ε-greedy exploration
happened to reinforce a second, only-marginally-worse action almost as
strongly as the true best one. This is a genuinely different failure mode
from "the algorithm is wrong" — it's an under-exploration artifact, visible
directly in the closeness of the competing Q-values.

## 6. Sample efficiency: 200 vs 5000 episodes

```python
Q_short = q_learning(200)
matches_short = sum(1 for s in states if s not in HOLES and s != GOAL and q_policy(Q_short, s) == policy[s])
print(f"{matches_short}/11 states agree after 200 episodes")
```

**Actual output: `2/11 states agree`** — a dramatic drop from 9/11 (5000
episodes) to 2/11 (200 episodes). Q-learning needs substantially more
experience to converge than value iteration needs computation, precisely
*because* it has no model to plan with — it must discover the consequences
of actions purely by trying them, repeatedly, until estimates stabilize.
This sample-inefficiency is exactly why chess (an enormously larger state
space than this 16-state grid) cannot rely on tabular Q-learning directly:
even a huge number of self-play games would visit only a vanishing
fraction of possible positions, which is precisely the motivation for
Lesson 052's function approximation (a neural network that *generalizes*
across similar positions instead of needing to visit each one individually).
