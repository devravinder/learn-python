# Reference Solution

```bash
python data/generate_data.py
python analysis.py
```

- [analysis.py](analysis.py) — EDA, TF-IDF pipeline, TF-IDF+SVD dense
  embedding pipeline, generalization stress test
- [FINDINGS.md](FINDINGS.md) — verified in-distribution + generalization
  results, including a real, honest failure mode (neutral human text
  flagged as AI) that mirrors an actual documented problem with real
  AI-text detectors

Try [01_requirement.md](../01_requirement.md) yourself first — in
particular, don't stop at in-distribution accuracy (it will look
deceptively perfect); the stress test in Q6 is where this project's real
lesson lives.
