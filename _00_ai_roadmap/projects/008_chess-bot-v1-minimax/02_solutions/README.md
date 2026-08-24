# Reference Solution

```bash
pip install python-chess
python chess_bot.py               # play interactively (you're White)
python chess_bot.py --benchmark    # Q3, Q5, Q6 checks
```

- [chess_bot.py](chess_bot.py) — negamax + alpha-beta search, material +
  piece-square-table + mobility evaluation, capture-first move ordering,
  interactive CLI, and the benchmark/sanity-check suite from
  [01_requirement.md](../01_requirement.md)
- [FINDINGS.md](FINDINGS.md) — expected results and what to check if your
  bot fails the free-queen sanity check (not independently verified here —
  no `python-chess`/internet access in the authoring sandbox; see the note
  at the top)

Try [01_requirement.md](../01_requirement.md) yourself first. This exact
`negamax`/`evaluate`/`order_moves` structure is what Project 009 reuses
almost unchanged, swapping only `evaluate()` for a trained neural network —
worth keeping your code modular along that exact boundary now.
