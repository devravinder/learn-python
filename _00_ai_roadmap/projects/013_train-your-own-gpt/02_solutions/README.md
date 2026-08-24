# Reference Solution

```bash
python data/generate_fallback_corpus.py     # or supply your own real text as data/*.txt
python train.py --data data/fallback_corpus.txt --vocab_size 512 --max_steps 3000
python generate.py --prompt "The forest spirit" --temperature 0.8 --max_new_tokens 100
```

- [bpe_tokenizer.py](bpe_tokenizer.py) — Lesson 068a's tokenizer, reused
  directly, with `save`/`load` added for reuse across runs
- [model.py](model.py) — Lessons 059-060's `GPT`, unchanged
- [train.py](train.py) — Lessons 064-065/067's data pipeline + training
  loop, plus a Lesson 068 sizing sanity-check printout
- [generate.py](generate.py) — Lesson 066's sampling (temperature, top-k,
  top-p)
- [FINDINGS.md](FINDINGS.md) — **verified** tokenizer training/compression
  numbers and sizing arithmetic (actually run); training-run
  loss/generation results are described as expected behavior only — no
  PyTorch in the authoring sandbox. Run it yourself for real numbers.

**Strongly consider swapping in a real text corpus** (a public-domain
novel, a collection of your own writing, anything with real linguistic
richness) instead of the synthetic fallback — `FINDINGS.md` explains
exactly why the fallback corpus's repetitiveness makes it a weak test of
what the architecture can actually do.

Try [01_requirement.md](../01_requirement.md) yourself first, in
particular the sizing exercise (Q1) — decide your model size from your
corpus's token count *before* picking architecture hyperparameters, the
same order of operations `train.py` uses.
