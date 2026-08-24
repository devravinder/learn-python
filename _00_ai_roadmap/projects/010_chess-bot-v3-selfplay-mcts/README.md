# Project 010 — Chess Bot v3: Self-Play RL + MCTS (AlphaZero-Style)

**Builds on lessons:** [050](../../lessons/050_intro-reinforcement-learning/README.md)–[054](../../lessons/054_self-play-alphazero/README.md) (Module 8 capstone)
**Difficulty:** Advanced (capstone-lite)
**Estimated time:** 8–12 hours (plus self-play training time, which scales with how many games/iterations you choose to run)

## Objective

The Module 8 capstone: a combined policy+value network, trained entirely
through self-play guided by MCTS, with **zero hand-crafted evaluation and
zero external game data** — the actual AlphaZero algorithm, implemented at
a scale that runs on a personal computer.

## Read this before you start: a realistic scope note

AlphaZero's original chess training used thousands of TPUs for hours.
Lesson 054 said this explicitly, and it's worth repeating here: **this
project's success criterion is demonstrating that the self-play loop
genuinely improves the network generation-over-generation** (measured
directly — see Q6), not "beats Project 008/009" or "plays strong chess."
Expect to run a small number of self-play iterations with a modest number
of games and MCTS simulations per move; the point is implementing and
proving out the *mechanism* correctly, exactly as Lesson 054's tic-tac-toe-
scale exercise did for the value-network half of this idea.

## Contents

1. [01_requirement.md](01_requirement.md)
2. [02_solutions/](02_solutions/)

## Dependency

`pip install python-chess`, plus `torch` (root `requirements.txt`).
