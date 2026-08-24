# Project 012 — Fine-Tuned Transformer Text Detector

**Builds on lessons:** [055](../../lessons/055_text-preprocessing-tokenization/README.md)–[062](../../lessons/062_tokenization-for-llms/README.md) (Module 10 capstone), and Project 011
**Difficulty:** Advanced
**Estimated time:** 4–6 hours (plus fine-tuning time)

## Objective

Upgrade Project 011's classical TF-IDF/embedding pipeline to a **fine-tuned
pretrained Transformer** (DistilBERT) on the exact same human-vs-AI-text
task. The real deliverable is the **comparison**: does contextual attention
actually fix the specific failure mode Project 011 documented (plain,
neutral human text getting flagged as AI)?

## Contents

1. [01_requirement.md](01_requirement.md)
2. [02_solutions/](02_solutions/)

## Dependency

```bash
pip install transformers datasets
```

(`torch` is already in the root `requirements.txt`.)
