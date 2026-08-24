# 02 — Practicals: Environment & Python Setup

Work through these in order. Don't peek at `03_solutions.md` until you're stuck or done.

1. **Create and activate a virtual environment** for this repo at its root
   (`learn-ai-ml/.venv`).

2. **Install the shared dependencies** from the root `requirements.txt` into that
   environment.

3. **Verify the install** by printing:
   - your Python version
   - the installed versions of `numpy`, `pandas`, `matplotlib`, `sklearn`, `torch`

4. **Launch Jupyter** (either `notebook` or `lab`) and create a new notebook at
   `lessons/001_environment-and-python-setup/scratch.ipynb` that:
   - imports `numpy` as `np`
   - creates an array `[1, 2, 3, 4, 5]`
   - prints its `.sum()` and `.mean()`

5. **Write a plain script** `lessons/001_environment-and-python-setup/check_env.py`
   that does the same version-printing as step 3, so you can re-run it any time
   with `python check_env.py` instead of opening Jupyter.
