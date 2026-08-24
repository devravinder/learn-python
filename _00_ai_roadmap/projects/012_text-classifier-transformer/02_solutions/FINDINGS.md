# Findings — Fine-Tuned Transformer Text Detector

*(Caveat: `transformers`/PyTorch execution wasn't possible in the
authoring sandbox — the analysis below is grounded reasoning about
expected behavior, not verified output. Run `finetune_distilbert.py`
yourself and replace this with your real results.)*

## In-distribution accuracy (expected)

DistilBERT fine-tuned for a few epochs should reach **~100% test
accuracy** on this dataset just as easily as Project 011's classical
pipeline did — the underlying task (distinguishing two clearly different
writing registers) isn't hard for either approach when evaluated on data
drawn from the same templates as training.

## The real question: does it fix "The weather today is nice"?

This is genuinely **uncertain without running it**, and worth treating as
a real experiment rather than assuming the "bigger model" wins:

- **The case for improvement**: DistilBERT's pretrained knowledge (from
  masked language modeling on a huge, diverse text corpus before this
  project ever fine-tunes it) already encodes that plain, neutral
  sentences like "The weather today is nice" are completely ordinary,
  common human sentences — a signal the classical TF-IDF/Naive-Bayes
  pipeline, trained *only* on this project's small templated dataset, has
  no way to access at all. Fine-tuning only needs to *nudge* that
  pretrained knowledge toward this specific task, not build a notion of
  "ordinary sentence" from scratch.
- **The case against improvement**: fine-tuning on a small, narrow, highly
  templated dataset (500 examples from ~16 templates) can just as easily
  **overwrite** that useful pretrained knowledge with the same narrow
  "informality = human" heuristic Project 011's classical model learned —
  a well-documented risk called **catastrophic forgetting**, especially
  likely when fine-tuning data doesn't include any examples resembling the
  edge case you care about (this dataset has no neutral-register human
  examples at all, in either the classical or Transformer version).

**The honest prediction**: without deliberately adding some neutral/formal
human examples to the training data, there's a real chance the fine-tuned
Transformer **still gets this wrong**, for the same fundamental reason
Project 011's model did — neither model has ever seen a neutral human
example labeled correctly. This would be a legitimate, interesting
finding: swapping in a more powerful architecture doesn't fix a *data*
problem by itself.

## What would actually fix it

If the stress test still fails after fine-tuning, the fix isn't a bigger
model — it's **better training data**: add genuinely neutral, formal, and
non-native-English-style human examples to the training set, so the model
has something to learn the boundary from. This is a real, general lesson
about ML systems: architecture upgrades (TF-IDF → Transformer) address
*representational* limitations (Lesson 057's static-embedding problem);
they do not substitute for *data coverage* of the cases you actually care
about.

## Reflection: what DistilBERT has that TF-IDF doesn't

TF-IDF represents "weather," "today," "nice" as three independent,
context-free dimensions with no relationship to each other or to any
notion of "ordinary sentence." DistilBERT's attention layers (Lesson 058)
produce **contextual** representations — informed by pretraining on
massive amounts of real text, it has effectively "seen" millions of plain,
unremarkable human sentences and can represent this one as unremarkable
too, *if* its fine-tuning doesn't override that. This is the theoretical
advantage; whether it manifests in practice depends entirely on whether
fine-tuning preserves or destroys that pretrained signal for this
particular narrow task — exactly what running the actual experiment
would tell you.
