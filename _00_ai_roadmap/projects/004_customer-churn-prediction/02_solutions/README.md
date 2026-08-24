# Reference Solution

```bash
python data/generate_data.py
python analysis.py
```

- [analysis.py](analysis.py) — EDA, preprocessing (with scaling only where
  needed), 4-model comparison, ROC comparison chart, feature importance
- [FINDINGS.md](FINDINGS.md) — verified EDA numbers, metric justification,
  and a business recommendation

Try [01_requirement.md](../01_requirement.md) yourself first — in
particular, decide your metric/threshold *before* looking at which model
wins, so you're not unconsciously picking the metric that flatters your
favorite model.
