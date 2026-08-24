# 02 — Practicals: Python Fundamentals & Data Structures

Write these as a script or notebook under this lesson's folder
(e.g. `lessons/002_python-fundamentals-and-data-structures/practice.py`).

1. Given `words = ["the", "cat", "sat", "on", "the", "mat"]`, build a `dict`
   mapping each **unique** word to an integer id, in first-seen order (this is a
   tiny vocabulary — exactly what you'll build for real before feeding text into a
   model).

2. Using a comprehension, turn `words` into a list of ids using the vocabulary from
   step 1.

3. Write a function `normalize(values, low=0.0, high=1.0)` that rescales a list of
   numbers into `[low, high]`. Test it on `[10, 20, 30, 40]`.

4. Write a generator function `batch(iterable, size)` that yields chunks of the
   given size (last chunk may be smaller). Use it to print `range(23)` in
   batches of 5.

5. Write a `Dataset` class with `__init__(self, samples, labels)`, `__len__`, and
   `__getitem__`. Construct one from `samples = ["a", "b", "c"]`,
   `labels = [0, 1, 0]`, then print `len(ds)` and `ds[1]`.

6. What happens if `values` passed to `normalize` all have the same value (so
   `vmax - vmin == 0`)? Fix `normalize` to handle that case without crashing.
