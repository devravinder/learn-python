# Reference Solution

```bash
pip install python-chess
python self_play.py --generations 3 --games 20 --sims 100     # slow - real self-play compute
python benchmark_vs_v1.py --generation 3 --games 10 --sims 200
```

- [network.py](network.py) — `(12,8,8)` board encoding, combined
  policy+value CNN, `(from,to)` move encoding (4096 actions, queen-only
  promotion)
- [mcts.py](mcts.py) — PUCT-based, network-guided MCTS (no random
  rollouts — Lesson 054's upgrade to Lesson 053)
- [self_play.py](self_play.py) — self-play game generation, combined-loss
  training, generation-vs-generation-0 win rate measurement
- [benchmark_vs_v1.py](benchmark_vs_v1.py) — v3 vs Project 008's classical
  bot, imported directly (no code duplication)
- [FINDINGS.md](FINDINGS.md) — realistic expected results at this training
  scale, and what to check if generation-over-generation improvement
  doesn't show up (not independently verified — see the note at the top)

**This will actually take real time to run** — self-play with MCTS is
inherently compute-heavy (each move requires many network forward passes).
Start with small numbers (`--games 5 --sims 30`) to confirm the pipeline
runs end-to-end and produces sane-looking output before committing to a
longer run with the defaults or larger settings.

Try [01_requirement.md](../01_requirement.md) yourself first, and read the
scope note in the project [README.md](../README.md) before you start —
it'll save you from judging your own results against the wrong bar.
