# Reference Solution

```bash
python data/generate_data.py
python analysis.py
```

- [analysis.py](analysis.py) — TF-IDF + Naive Bayes/Logistic Regression
  pipeline, threshold tuning, error analysis, feature inspection, applied to
  both datasets
- [FINDINGS.md](FINDINGS.md) — verified results, including a hand-written
  generalization test that reveals a real, honest limitation of bag-of-words
  models (not just in-distribution metrics, which are misleadingly perfect
  here)

Try [01_requirement.md](../01_requirement.md) yourself first. If your model
also hits ~100% test accuracy, don't assume something's wrong — instead, do
what `FINDINGS.md` does: test it on sentences you write yourself that don't
match any training template, and see where it actually breaks.
