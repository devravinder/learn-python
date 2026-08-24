# Assignment 003 — Backprop + a 2-Layer NN From Scratch (NumPy, Vectorized)

**Covers lessons:** [035](../../lessons/035_perceptron-mlp/README.md)–[038](../../lessons/038_nn-from-scratch/README.md)
**Estimated time:** 2–3 hours

## Objective

Lesson 038 built a scalar-by-scalar autograd engine (micrograd-style). This
assignment builds the **vectorized, matrix-form** version instead — the
`∂L/∂W = x.T @ (∂L/∂z)` equations from Lesson 037's last section — which is
what real frameworks actually compute under the hood for speed. No
autograd library, no `Value` class: raw NumPy matrix operations, gradients
derived and coded by hand.

## Contents

1. [01_questions.md](01_questions.md)
2. [02_solutions/](02_solutions/)
