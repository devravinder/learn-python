# Findings — Chess Bot v1

*(Caveat: `python-chess` isn't installed in the sandbox this curriculum was
authored in, and no internet/pip access was available to install it — so
unlike Projects 001–005 and 048/049's tic-tac-toe code (all pure-stdlib and
actually executed), the numbers below are **expected behavior**, not
verified output. Run `chess_bot.py --benchmark` yourself and replace this
with your real results.)*

## Move ordering (expected)

Trying captures first should noticeably reduce node count at a fixed depth
compared to unordered search — the same effect measured concretely in
Lesson 049 on tic-tac-toe (94% reduction from pruning alone; move ordering
on top of that typically yields a further 30-70% reduction depending on
position, since finding a strong move early lets alpha-beta cut off more
branches sooner).

## Depth vs depth self-play (expected)

Depth 4 should win the large majority of games against depth 2 — chess
strength scales strongly with search depth even with an identical, simple
evaluation function, since deeper search catches more tactics (forks,
skewers, short combinations) the evaluation function alone can't see
statically. Expect deep search to take roughly `35x` longer per move than
one ply shallower (chess's branching factor is roughly 35 legal moves per
position on average) *before* pruning; alpha-beta with capture-ordering
should keep the real slowdown well below that, but still substantial —
this exact tradeoff (strength vs speed per additional ply) is the central
engineering constraint of classical chess engines.

## Sanity checks (expected)

The free-queen-capture position is deliberately trivial (an undefended
queen one square away on an open file) — even depth-1 search should find
`Qxd8` immediately, since it's a direct material gain visible in the very
next evaluation. If your bot fails this check, look first at `order_moves`
and `evaluate` — a common bug is an inverted sign somewhere in the
White/Black perspective conversion, which this test is specifically
designed to catch early and cheaply, before debugging anything more subtle.

## What Project 008 does and doesn't do well

This bot should already play *reasonable* amateur-level chess (competent
material play, avoids obvious blunders at depth 3+) but will miss
positional subtleties no simple material+PST+mobility evaluation can
capture — poor pawn structure, long-term king safety, subtle piece
coordination. That gap is exactly what Project 009's learned CNN evaluation
is built to close.
