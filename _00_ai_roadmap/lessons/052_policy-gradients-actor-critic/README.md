# Lesson 052 — Policy Gradients (REINFORCE) & Actor-Critic Basics

**Module:** 8 — Reinforcement Learning & Game AI (→ Chess Bot)
**Prerequisites:** [036](../036_activation-functions-softmax/README.md), [051](../051_q-learning-value-iteration/README.md)
**Estimated time:** 2.5 hours

## Objective

Q-learning (Lesson 051) learns *values*, then derives a policy from them.
This lesson learns a **policy directly** — a probability distribution over
actions, adjusted via gradient ascent on expected reward. This is the
family of methods (extended in Lesson 054) that self-play chess engines
like AlphaZero are actually trained with.

## Contents

1. [01_concepts.md](01_concepts.md)
2. [02_practicals.md](02_practicals.md)
3. [03_solutions.md](03_solutions.md)

## Resources

- [David Silver — RL Lecture 7: Policy Gradient Methods](https://www.youtube.com/watch?v=KHZVXao4qXs)
