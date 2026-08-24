# Project 009 — Chess Bot v2: CNN Position Evaluator

**Builds on lessons:** [039](../../lessons/039_pytorch-fundamentals/README.md)–[044](../../lessons/044_cnn-architectures/README.md), [048](../../lessons/048_game-trees-minimax/README.md)–[049](../../lessons/049_alpha-beta-pruning-evaluation/README.md)
**Difficulty:** Advanced
**Estimated time:** 5–7 hours

## Objective

Keep Project 008's search algorithm (negamax + alpha-beta, unchanged) and
replace only its hand-crafted `evaluate()` function with a **trained CNN**.
Since no labeled human-game dataset or existing chess engine is assumed
available, this project uses **knowledge distillation**: train the CNN to
imitate Project 008's own hand-crafted evaluation function, learned purely
from board positions and the classical evaluation's output — a real,
legitimate ML technique (distilling a slower/hand-built function into a
fast learned approximator), not a shortcut.

## Contents

1. [01_requirement.md](01_requirement.md)
2. [02_solutions/](02_solutions/)

## Dependency

Same as Project 008: `pip install python-chess`, plus `torch` (already in
the root `requirements.txt`).
