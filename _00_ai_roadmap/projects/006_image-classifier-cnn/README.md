# Project 006 — Image Classifier (Intro CNN)

**Builds on lessons:** [039](../../lessons/039_pytorch-fundamentals/README.md)–[042](../../lessons/042_regularization-dropout-batchnorm/README.md) (Module 6 capstone)
**Difficulty:** Intermediate
**Estimated time:** 3–4 hours

## Objective

Your first image classifier, and your first hands-on use of `nn.Conv2d` —
full CNN theory (why convolutions work, pooling, architectures) is Module
7's Lesson 043, but building one practically here first, then learning the
theory, mirrors how most people actually learn this (use the tool, then
understand why it works). Compares a plain MLP against a small CNN on the
same image data to make the difference concrete.

## Contents

1. [01_requirement.md](01_requirement.md)
2. [02_solutions/](02_solutions/)

## Data

`sklearn.datasets.load_digits()` — 1797 handwritten digit images (8x8
grayscale, 10 classes), built into scikit-learn, no download required.
