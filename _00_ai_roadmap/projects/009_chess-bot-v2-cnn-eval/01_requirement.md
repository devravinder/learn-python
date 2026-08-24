# 01 — Requirement: Chess Bot v2 (CNN Position Evaluator)

## The brief

Replace Chess Bot v1's hand-crafted `evaluate()` with a CNN trained to
predict position value — same search code, a *learned* leaf evaluation
instead of a hand-written formula.

## What to produce

1. **Board encoding**: represent a `chess.Board` as a `(12, 8, 8)` tensor —
   one 8x8 plane per (piece type, color) combination (6 piece types x 2
   colors), with `1.0` at a square if that piece/color occupies it, else
   `0.0`. This is the standard input representation used by real chess
   neural networks (including AlphaZero, with additional planes for
   history/castling rights that you can skip for this simplified version).

2. **Training data via distillation**: generate a large set of positions by
   playing random legal moves from the starting position for a random
   number of plies (e.g. 0-40), for maybe 5,000-20,000 positions. Label
   each position with Project 008's `evaluate()` function (the classical
   material + piece-square-table + mobility score). This gives you
   `(board_tensor, classical_eval_score)` pairs with no external dataset
   needed.

3. **CNN evaluator**: build a small CNN (Lessons 043-044 — a few conv
   layers + global pooling or a small FC head) that takes the `(12,8,8)`
   tensor and predicts a single scalar value. Train with MSE loss
   (Lesson 020) on the distillation dataset, holding out a validation
   split.

4. **Swap it into the search**: replace `evaluate(board)` in Project 008's
   `negamax` with `cnn_evaluate(board)` (run the board through the trained
   network, no gradient tracking needed at inference —
   `torch.no_grad()`). Everything else (negamax, alpha-beta, move
   ordering) stays exactly the same.

5. **Validate the distillation worked**: on the held-out validation set,
   report correlation (or R², Lesson 018) between the CNN's predictions and
   the classical evaluation's true values. A good distillation should show
   strong (not necessarily perfect) agreement.

6. **Head-to-head benchmark**: play Chess Bot v1 (classical eval) against
   Chess Bot v2 (CNN eval) at the same search depth, 10 games alternating
   colors. Since v2 is *trained to imitate* v1's evaluation, do you expect
   it to play similarly, slightly worse (imperfect distillation), or is
   there a chance it plays differently in interesting ways (the CNN
   smooths over/generalizes past the exact hand-crafted formula, possibly
   changing move choices in close positions)? Report what you actually
   observe.

7. **Reflection**: what did you actually gain from this exercise, given the
   CNN was trained to imitate a function you already had? (Hint: think
   about what changes if, instead of distilling from a hand-crafted
   function, you had access to millions of real labeled grandmaster games
   or engine-analyzed positions instead — what would the *exact same*
   training pipeline let you do that Project 008 fundamentally couldn't?)

## Constraints

- Keep the search code (negamax, alpha-beta, move ordering) unchanged from
  Project 008 — only the evaluation function changes. This isolates
  exactly what a learned evaluation buys you, with everything else held
  constant.
- Don't peek at `02_solutions/` before you have a trained, integrated CNN
  evaluator yourself.
