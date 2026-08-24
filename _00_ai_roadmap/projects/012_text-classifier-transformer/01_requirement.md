# 01 — Requirement: Fine-Tuned Transformer Text Detector

## The brief

Reuse Project 011's `human_vs_ai.csv`. Fine-tune a small pretrained
Transformer (`distilbert-base-uncased`) as a classifier, and directly
compare it against Project 011's classical baseline — especially on the
exact stress-test sentences Project 011 documented as failure cases.

## What to produce

1. **Tokenize with the model's own tokenizer** (Lesson 062 — DistilBERT
   uses WordPiece, not the TF-IDF pipeline's word-level tokenization):
   `AutoTokenizer.from_pretrained("distilbert-base-uncased")`.

2. **Fine-tune**: load `AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)`
   and fine-tune on the training split (a handful of epochs — this is a
   small, easy dataset, not a large corpus). Use
   `transformers.Trainer` or a manual PyTorch loop (Lesson 040) — either
   is fine, but understand what the `Trainer` is doing under the hood if
   you use it.

3. **Evaluate**: report `classification_report` (Lesson 024) on the same
   held-out test split Project 011 used, for direct comparability.

4. **The real test — rerun Project 011's exact stress-test sentences**:
   ```python
   novel_texts = [
       "honestly not sure how i feel about all this ai stuff, kinda weird ngl",
       "In summary, this topic warrants further investigation and analysis by researchers.",
       "The weather today is nice.",   # <- Project 011's classical model got this WRONG
       "This is a great product I really enjoyed using it every day.",
   ]
   ```
   Does the fine-tuned Transformer correctly classify **"The weather today
   is nice"** as human (fixing Project 011's documented failure), or does
   it make the same mistake? Report the actual result — either outcome is
   a valid, reportable finding.

5. **Try the "casual AI" adversarial case** from Project 011's harder
   tests too (an AI-style sentence written informally). Does the
   Transformer generalize any better here than the classical style-based
   detector, or does it share the same blind spot?

6. **Reflection**: given what you know about attention (Lesson 058) and
   contextual embeddings (Lesson 057's limitation that motivated it),
   explain *why* a fine-tuned Transformer might (or might not) handle the
   "neutral human text" failure case better than a bag-of-words style
   classifier — what information does DistilBERT have access to that
   TF-IDF fundamentally doesn't?

## Constraints

- Fine-tune, don't just use the pretrained model's existing knowledge
  zero-shot — the point is practicing the fine-tuning workflow
  (Lesson 040's training loop, applied to a pretrained model).
- Don't peek at `02_solutions/` before you've fine-tuned your own model and
  run the stress test yourself.
