# 01 — Concepts: Environment & Python Setup

## Why isolate environments at all

Every project you touch will want slightly different versions of NumPy, PyTorch, etc.
If you install packages globally, upgrading one project's dependency can silently
break another. A **virtual environment** is a private copy of the Python interpreter
plus its own `site-packages` folder, so `pip install` only affects the current project.

## Tooling choices

- **`venv`** (built into Python 3) — simplest option, no extra install. Good default
  for this repo.
- **`conda`/`mamba`** — heavier, but handles non-Python dependencies (CUDA, BLAS)
  well. Worth switching to once you're training PyTorch models on a GPU.
- **`uv`** — modern, very fast drop-in replacement for `pip`/`venv`. Fine to use if
  you already have it; commands below use plain `venv`/`pip` since they're universal.

## Core commands

```bash
# create an environment (once per machine/checkout)
python3 -m venv .venv

# activate it (every new shell session)
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

# install this repo's shared dependencies
pip install -r requirements.txt

# leave the environment
deactivate
```

Activating just changes `PATH` so `python`/`pip` point inside `.venv/`. You'll know
it worked because your shell prompt gets a `(.venv)` prefix.

## Jupyter

Jupyter gives you a notebook: cells you execute independently, with output (including
plots) inline. It's the standard tool for exploratory data work; `.py` scripts are
better once code needs to be reused or run unattended.

```bash
jupyter notebook      # classic UI
# or
jupyter lab           # newer UI, more features
```

## Verifying an install

Two useful one-liners you'll reuse constantly when debugging environment issues:

```bash
python -c "import sys; print(sys.version)"
python -c "import numpy, pandas; print(numpy.__version__, pandas.__version__)"
```

## Suggested project layout (what this repo already follows)

```
requirements.txt   shared dependencies
lessons/           theory + practice, one concept per folder
projects/          real-world builds
assignments/       checkpoint exercises
```

Each lesson/project is self-contained: you should be able to open just that folder
and know what to do from its `README.md`.
