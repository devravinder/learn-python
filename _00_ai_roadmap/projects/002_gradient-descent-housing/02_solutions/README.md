# Reference Solution

```bash
python data/generate_data.py
python analysis.py
```

- [analysis.py](analysis.py) — from-scratch batch GD, closed-form validation,
  learning-rate comparison, and mini-batch convergence-speed comparison, all
  in plain NumPy
- [FINDINGS.md](FINDINGS.md) — verified numeric results and interpretation

Try [01_requirement.md](../01_requirement.md) yourself first. The single most
useful debugging step if your gradient descent doesn't match the closed-form
solution: implement the numerical-gradient check from Lesson 013/014 on your
`gradients()` function before assuming the learning rate or epoch count is
the problem.
