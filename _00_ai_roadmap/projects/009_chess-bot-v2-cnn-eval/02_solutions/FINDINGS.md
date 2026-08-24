# Findings — Chess Bot v2 (CNN Evaluator)

*(Caveat, same as Project 008: `python-chess`/PyTorch execution wasn't
possible in the authoring sandbox — the numbers below are expected
behavior for this distillation setup, not verified output. Run the scripts
yourself and replace this with your real results.)*

## Distillation quality (expected)

A CNN with this capacity (3 conv layers + global average pooling + small
FC head) trained on 8,500 positions (after an 85/15 split) should achieve a
**validation R² well above 0.8** predicting the classical evaluation — the
task is comparatively easy for a neural network: material counting and
piece-square lookups are exactly the kind of local, translation-aware
pattern a CNN is built to detect (Lesson 043's core argument), and the
target function itself has no noise (the classical evaluation is
deterministic) — pure distillation of a fixed function is generally easier
than learning from noisy real-world labels.

## v1 vs v2 head-to-head (expected)

Since v2 is trained to *imitate* v1's evaluation rather than improve on it,
expect the two bots to perform **similarly** in the benchmark — roughly an
even split of wins, more draws than either "winning decisively," with
whatever edge exists likely coming down to small distillation
imperfections rather than a genuine strategic difference. **This is the
correct, expected outcome, not a disappointing one** — v2 cannot exceed
what it was trained to copy; any advantage over v1 would only come from the
CNN *generalizing/smoothing* in a way that occasionally helps in positions
slightly off-distribution from training data, not from learning anything
v1 didn't already encode.

## The honest reflection (Q7)

This project's real value isn't "v2 beats v1" (it structurally can't, by
construction) — it's proving the **mechanism**: a trained network can slot
into the exact same search code as a hand-crafted evaluation function, with
zero changes to `negamax`/`order_moves`. The moment you have a *better*
signal than your own hand-crafted formula to train against — real
grandmaster games labeled by outcome, or a strong existing engine's deep
analysis of millions of positions — this exact same pipeline
(board-to-tensor, CNN, MSE training, swap into search) lets the evaluation
function **exceed** anything you could hand-craft, which is precisely how
real modern engines (Stockfish's NNUE, Leela Chess Zero) work. Project 009
proves the plumbing works before Project 010 changes what's flowing through
it — from "imitate a fixed formula" to "improve via self-play."
