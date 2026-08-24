# Reference Solution

```bash
pip install transformers datasets
python finetune_distilbert.py
```

- [finetune_distilbert.py](finetune_distilbert.py) — tokenization,
  `Trainer`-based fine-tuning, evaluation, and Project 011's exact
  stress-test sentences rerun on the fine-tuned model
- [FINDINGS.md](FINDINGS.md) — an honestly *uncertain* prediction (not
  independently verified — no `transformers`/PyTorch execution in the
  authoring sandbox) about whether fine-tuning fixes Project 011's
  documented failure case, with reasoning for both outcomes and what
  "catastrophic forgetting" would mean here

Try [01_requirement.md](../01_requirement.md) yourself first. **This
project's most valuable outcome is genuinely not knowing the answer in
advance** — run the stress test, see what actually happens, and if "The
weather today is nice" is still misclassified, that's real data about the
limits of architecture upgrades without data upgrades, not a failed
project.
