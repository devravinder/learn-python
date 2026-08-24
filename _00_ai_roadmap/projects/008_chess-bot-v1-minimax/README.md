# Project 008 — Chess Bot v1: Minimax + Alpha-Beta

**Builds on lessons:** [048](../../lessons/048_game-trees-minimax/README.md)–[049](../../lessons/049_alpha-beta-pruning-evaluation/README.md)
**Difficulty:** Intermediate
**Estimated time:** 4–6 hours

## Objective

Your first real Chess Bot — **zero machine learning involved**. Everything
here is classical search (Lesson 048) plus a hand-crafted evaluation
function (Lesson 049), exactly like real chess engines worked for decades
before neural network evaluation became common. Project 009 replaces the
hand-crafted evaluation with a trained CNN; Project 010 replaces the search
strategy itself with self-play reinforcement learning + MCTS. This version
is the baseline both later versions are compared against.

## Contents

1. [01_requirement.md](01_requirement.md)
2. [02_solutions/](02_solutions/)

## Dependency

This project uses [`python-chess`](https://python-chess.readthedocs.io/)
for board representation, legal move generation, and rules (check, castling,
en passant, checkmate/stalemate detection) — reimplementing full chess
rules from scratch is out of scope; the point of this project is the search
algorithm and evaluation function, which you build entirely yourself.

```bash
pip install python-chess
```
