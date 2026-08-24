# Findings — Chess Bot v3 (Self-Play + MCTS)

*(Caveat, same as Projects 008-009: `python-chess`/PyTorch execution wasn't
possible in the authoring sandbox, and this project in particular involves
enough compute — self-play games x MCTS simulations x network forward
passes — that it wasn't just skipped for lack of libraries but would be
genuinely slow to run without a real machine. Every number below is
expected behavior grounded in how AlphaZero-style training is documented
to behave, not verified output. Run `self_play.py` yourself, expect it to
take real wall-clock time even at this reduced scale, and replace this file
with your actual results.)*

## What "success" looks like at this scale

The **only** number this project should be judged on is the generation-vs-
generation win rate (`self_play.py`'s final printout each generation). At
this tutorial scale (tens of games, ~100 MCTS simulations per move, a
handful of generations), expect:

- **Generation 1 vs Generation 0 (random init)**: a clearly-above-50% win
  rate — even a small amount of self-play + training should let the
  network learn basic material safety (not hanging pieces for free) that a
  randomly-initialized network has no notion of at all.
- **Generation 2-3 vs Generation 0**: win rate should climb further, though
  possibly with noisy, non-monotonic steps between individual generations
  (self-play at this small a scale is a high-variance process — 20 games
  is not a lot of data to learn from or to measure improvement with
  precisely).

If win rate does **not** climb above roughly 50-60% by generation 2-3,
the most likely causes, in order of likelihood, are: too few MCTS
simulations per move (the policy/value targets are too noisy to learn
from), too few self-play games per generation (same issue, different
source), or a bug in the perspective/sign handling in `backpropagate` or
the training targets (Lesson 053's solutions document a real example of
exactly this class of bug, caught by a similar "should never lose to a
weak baseline" sanity check).

## v3 vs Project 008's classical bot (expected)

**Do not expect Generation 3 to beat Project 008's v1** at this training
scale. AlphaZero's original chess run used on the order of 5,000 TPUs for
about 9 hours (roughly tens of millions of self-play games) to reach
superhuman strength — many orders of magnitude beyond what tens of
self-play games across 3 generations can produce. Project 010's v3 at this
scale should be expected to lose the majority of games against even a
shallow-depth (depth 3) classical bot — the classical bot's exhaustive
depth-3 tactical search still reliably outperforms a barely-trained
network's pattern recognition at this data scale.

## What this project actually demonstrates

The measurable, honest success criterion is the **mechanism working
end-to-end and improving monotonically-ish over generations** — network
learns from its own self-play data, gets measurably better than its
own earlier version, with zero hand-crafted evaluation and zero external
game data anywhere in the loop. That mechanism, scaled up with
vastly more compute (more simulations per move, more self-play games per
generation, more generations, a bigger network) and nothing else changed,
is *exactly* what produced AlphaZero's superhuman chess play — the
difference between this project's result and a superhuman engine is
compute scale, not a missing algorithmic idea.
