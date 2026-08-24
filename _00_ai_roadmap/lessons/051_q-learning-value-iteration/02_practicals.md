# 02 — Practicals: Q-Learning & Value Iteration

A 4x4 gridworld (states numbered 0-15, row-major): agent starts at state 0
(top-left), goal at state 15 (bottom-right, reward `+1`, terminal), holes
at states `5, 7, 11, 12` (reward `-1`, terminal). Actions: up/down/left/
right; moving into a wall just keeps you in place.

```
. . . .
. H . H
. . . H
H . . G
```

1. Implement `step(state, action) -> (next_state, reward, done)` for this
   grid (moves that would leave the grid are clamped to stay in bounds).

2. Implement `value_iteration()` (per `01_concepts.md`) for this MDP (you
   know the transition function exactly — it's what you just wrote in Q1 —
   so this is the model-based setting). Print the resulting `V(s)` values
   laid out as a 4x4 grid. Do they increase monotonically as you get closer
   to the goal, avoiding holes?

3. Extract the optimal policy from your converged `V` (pick, for each
   state, the action maximizing `r + γ*V(next_state)`). Print it as a grid
   of arrows. Does the policy visibly route around the holes?

4. Implement Q-learning (per `01_concepts.md`) with ε-greedy action
   selection, training for 5000 episodes. Extract the greedy policy from
   the learned `Q` table (`argmax_a Q(s,a)` for each state) and compare it
   to value iteration's policy from Q3 — what fraction of states agree?

5. Investigate a mismatch (if any): for a state where Q-learning's policy
   differs from value iteration's, print `Q(s, a)` for all 4 actions at
   that state. Is it a close call (similar Q-values, plausibly still
   under-explored) or a clear mistake?

6. Reduce training to 200 episodes instead of 5000. Does the Q-learning
   policy's agreement with the true optimal policy (Q3) get noticeably
   worse? What does this tell you about the sample-efficiency tradeoff
   between "compute the exact answer via a known model" (value iteration)
   and "learn an approximate answer purely from experience" (Q-learning) —
   and why real-world problems (including chess) usually can't use value
   iteration directly?
