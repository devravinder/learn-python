# Reference Solution

```bash
python data/generate_data.py    # writes data/retail_sales.csv (stdlib only)
python analysis.py               # data quality report + charts/ + prints stats
```

- [analysis.py](analysis.py) — full pandas/matplotlib/seaborn pipeline
- [FINDINGS.md](FINDINGS.md) — the stakeholder-facing write-up
- `charts/` — generated on run (git-ignored scratch output, not committed)

Try [01_requirement.md](../01_requirement.md) yourself first — in particular,
think about *why* each cleaning decision is made, not just what the numbers
come out to. There's more than one defensible way to handle the missing/
negative rows; `analysis.py` documents the reasoning behind the choices it
makes so you can compare against your own.
