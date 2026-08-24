# 02 — Practicals: MDPs, Reward, Policy, Value

## Pen-and-paper / conceptual

1. For a reward sequence `[1, 1, 1, 1, 1]` (constant reward of 1 for 5
   steps), compute the return `G_0` for `γ = 1.0`, `γ = 0.9`, and
   `γ = 0.5`. Explain in words why `γ=1.0` gives exactly 5 while the others
   give less.

2. Define states, actions, and rewards for tic-tac-toe as an MDP (you
   already have all the game logic from Lesson 048): what is a state? What
   are the actions available in a state? What is the reward, and *when* is
   it received (every move, or only at the end)? Is the transition function
   deterministic or stochastic?

## Code: a tiny gridworld MDP

A 1D "corridor" MDP: 5 states in a row (`0,1,2,3,4`), agent starts at state
`2`. Actions: move left (`-1`) or right (`+1`). Reaching state `4` gives
reward `+1` and ends the episode; reaching state `0` gives reward `-1` and
ends the episode; every other move gives reward `0`.

3. Implement the environment: `step(state, action) -> (next_state, reward, done)`.

4. Implement a **uniformly random policy** (50/50 left/right) and run 1000
   full episodes (from state `2` until `done`), recording the total return
   `G_0` of each episode (`γ=1.0` since episodes are short and finite).
   Estimate `V(2)` under the random policy as the average return across all
   1000 episodes — this is **Monte Carlo policy evaluation**, estimating a
   value function purely from sampled experience rather than solving the
   Bellman equation directly.

5. Repeat Q4 but with a **right-biased** policy (70% right, 30% left —
   under a purely symmetric 50/50 policy the value is ≈0 either way, which
   hides the effect), comparing `γ=1.0` against `γ=0.9`. Explain the
   direction of the difference (should discounting *future* reward from a
   win that's now more likely, but still a few steps away, change the
   value estimate up or down?).

6. Change the policy to always move right (deterministic). Compute
   `V(2)` exactly by hand (no need to simulate — it's deterministic) for
   `γ=1.0` and confirm your simulated Monte Carlo estimate from a biased
   random policy (e.g. 80% right, 20% left) trends toward this value as the
   right-bias increases.
