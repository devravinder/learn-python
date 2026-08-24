# 03 — Solutions: MDPs, Reward, Policy, Value

## 1. Return at different discount factors

```
γ=1.0: G_0 = 1+1+1+1+1 = 5.0
γ=0.9: G_0 = 1 + 0.9 + 0.81 + 0.729 + 0.6561 = 4.0951
γ=0.5: G_0 = 1 + 0.5 + 0.25 + 0.125 + 0.0625 = 1.9375
```

`γ=1.0` weights every future reward exactly as much as immediate reward, so
the return is simply the sum. Smaller `γ` shrinks each successive term
geometrically, so rewards further in the future contribute almost nothing
— by step 5, a `γ=0.5` agent barely "feels" that reward at all.

## 2. Tic-tac-toe as an MDP

- **State**: the current board configuration (9 cells, each `X`/`O`/empty)
  plus whose turn it is.
- **Actions**: indices of empty cells (legal moves) from the current state.
- **Reward**: `0` for every non-terminal move; `+1`/`-1`/`0` (win/loss/draw)
  only at the **final** move of the game — a **sparse, delayed reward**,
  which is exactly why Lesson 048's minimax (which searches all the way to
  terminal states) sidesteps the "how do you learn from delayed reward"
  problem entirely, while an RL approach (Lesson 054) has to solve it via
  the discount factor and value functions.
- **Transitions**: deterministic — a move fully determines the next board
  state, matching the Markov property directly.

## 3. Gridworld environment

```python
def step(state, action):
    next_state = state + action
    if next_state == 4:
        return next_state, 1, True
    if next_state == 0:
        return next_state, -1, True
    return next_state, 0, False
```

## 4. Monte Carlo policy evaluation, random policy

```python
import random

def run_episode(policy_right_prob, gamma):
    state = 2
    rewards = []
    done = False
    while not done:
        action = 1 if random.random() < policy_right_prob else -1
        state, reward, done = step(state, action)
        rewards.append(reward)
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
    return G

random.seed(0)
returns = [run_episode(0.5, 1.0) for _ in range(1000)]
print(sum(returns) / len(returns))
```

**Actual output: `-0.032`** — close to 0, as expected: a perfectly
symmetric random walk from the middle state is equally likely to end at
`+1` or `-1`, so the average return should converge toward 0 as sample size
grows (small deviations like `-0.032` are just Monte Carlo sampling noise
at 1000 episodes).

## 5. Effect of discounting under a biased policy

```python
random.seed(0)
returns_g1 = [run_episode(0.7, 1.0) for _ in range(2000)]
print("gamma=1.0:", sum(returns_g1) / len(returns_g1))

random.seed(0)
returns_g09 = [run_episode(0.7, 0.9) for _ in range(2000)]
print("gamma=0.9:", sum(returns_g09) / len(returns_g09))
```

**Actual output: `gamma=1.0: 0.679`, `gamma=0.9: 0.538`.** With a
right-biased (70%) policy, the agent usually reaches the `+1` reward, but
it takes a few steps to get there — discounting (`γ=0.9`) shrinks that
future `+1` before it "counts," so the estimated value drops noticeably
(from 0.679 to 0.538) even though the *policy* (and thus the actual
win rate) hasn't changed at all. This is the discount factor's effect in
isolation: it changes how much a future reward is worth *today*, not
whether the reward happens.

## 6. Deterministic always-right policy, by hand

From state 2, always moving right: `2 -> 3` (reward 0) `-> 4` (reward +1),
episode ends. `G_0 = 0 + γ*1 = γ`. For `γ=1.0`, `V(2) = 1.0` exactly — a
guaranteed win in 2 moves with no discounting.

```python
random.seed(1)
for bias in [0.5, 0.6, 0.7, 0.8, 0.9, 0.99]:
    returns_bias = [run_episode(bias, 1.0) for _ in range(2000)]
    print(bias, sum(returns_bias) / len(returns_bias))
```

**Actual output:**

```text
0.5:  0.001
0.6:  0.382
0.7:  0.700
0.8:  0.870
0.9:  0.976
0.99: 0.999
```

As the right-bias approaches 1.0 (i.e. the policy approaches the
deterministic always-right policy), the Monte Carlo value estimate climbs
smoothly toward the hand-computed exact value of `1.0` — a direct,
verified confirmation that Monte Carlo policy evaluation converges to the
true value function as the policy (and sample size) approach the case
you can verify by hand.
