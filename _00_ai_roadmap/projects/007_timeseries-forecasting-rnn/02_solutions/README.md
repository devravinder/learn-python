# Reference Solution

```bash
python data/generate_data.py
python analysis.py
```

- [analysis.py](analysis.py) — EDA, 3 baselines, LSTM forecaster (scaler
  fit on train only — no test-set leakage), multi-step horizon error plot
- [FINDINGS.md](FINDINGS.md) — verified baseline numbers (seasonal-naive is
  a surprisingly strong bar to beat) + expected LSTM behavior

Try [01_requirement.md](../01_requirement.md) yourself first. Pay particular
attention to *not* leaking test-period statistics into your normalization —
`analysis.py` fits the scaler's mean/std using only the training portion of
the series, a detail that's easy to get wrong and would quietly inflate
your reported accuracy if missed.
