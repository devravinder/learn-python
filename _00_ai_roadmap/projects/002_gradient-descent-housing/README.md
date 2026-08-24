# Project 002 — Gradient Descent From Scratch on Real Housing Data

**Builds on lessons:** [010](../../lessons/010_linear-algebra-vectors/README.md)–[015](../../lessons/015_gradient-descent/README.md)
**Difficulty:** Basic–Intermediate
**Estimated time:** 3–4 hours

## Objective

Implement multiple linear regression's training loop entirely from scratch —
vectors, matrix operations, and gradient descent, no `sklearn.fit()` — on a
synthetic-but-realistic housing price dataset, then validate your from-scratch
result against the closed-form normal equation solution.

## Contents

1. [01_requirement.md](01_requirement.md)
2. [02_solutions/](02_solutions/)

## Data

`02_solutions/data/generate_data.py` (stdlib only) produces
`housing.csv` — square footage, bedrooms, age, distance to city center, and
sale price for 500 houses.
