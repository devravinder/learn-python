# Reference Solution

```bash
pip install python-chess
python generate_training_data.py     # ~10,000 labeled positions via distillation
python train_cnn_evaluator.py         # trains cnn_evaluator.pt
python chess_bot_v2.py                # play interactively (you're White)
python chess_bot_v2.py --benchmark     # distillation quality + v1 vs v2
```

- [generate_training_data.py](generate_training_data.py) — reproduces
  Project 008's classical evaluation (self-contained) to label randomly
  reached positions
- [train_cnn_evaluator.py](train_cnn_evaluator.py) — `(12,8,8)` board
  encoding, small CNN + global average pooling (Lesson 044), MSE training
- [chess_bot_v2.py](chess_bot_v2.py) — identical `negamax`/`order_moves`
  to Project 008, `evaluate()` swapped for the trained CNN
- [FINDINGS.md](FINDINGS.md) — expected results and why "v2 ties v1" is
  the *correct* outcome here, not a failure (not independently verified —
  no `python-chess`/PyTorch execution in the authoring sandbox)

Try [01_requirement.md](../01_requirement.md) yourself first, especially
Q7's reflection — it's the part of this project actually worth the effort.
