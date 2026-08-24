# Reference Solutions

```bash
python solutions.py
```

The `∂L/∂z2 = y_hat - y` simplification (used directly in `backward()`) was
verified against an independent pure-Python (no NumPy) implementation with
a numerical gradient check, matching to 8+ significant figures — see the
module docstring in [solutions.py](solutions.py).

With `n_hidden=2` (Q6), expect visibly worse separation of the two moons
than with `n_hidden=8` — a hands-on repeat of Lesson 035's "not enough
capacity" point, now on a real vectorized network you built and verified
yourself rather than a toy hand-picked-weights example.
