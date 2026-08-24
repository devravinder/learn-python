# Project 013 — Train a Small GPT on a Custom Corpus (Capstone-Lite)

**Builds on lessons:** [060](../../lessons/060_transformer-architecture/README.md), [063](../../lessons/063_language-modeling-objective/README.md)–[068a](../../lessons/068a_bpe-tokenizer-from-scratch/README.md) (Module 11 capstone)
**Difficulty:** Advanced (capstone-lite)
**Estimated time:** 6–10 hours (plus training time)

## Objective

The moment this entire curriculum has been building toward: train your own
GPT, from scratch, on text of your choosing, and watch it generate new
text. Every piece is something you already built yourself across Module
11 — this project's job is assembling them into one working pipeline, not
introducing anything new.

## What "your own LLM" means here, honestly

This is a real, working, from-scratch GPT — the same architecture (Lesson
060), same training objective (Lesson 063), same tokenizer approach
(Lesson 068a) as GPT-2. It is **not** GPT-2-scale (Lesson 068's scaling
laws explain exactly why that gap exists and roughly how large it is) — a
laptop-trainable model, sized appropriately for your actual corpus and
compute via Lesson 068's budgeting approach. Treat this as nanoGPT's scope,
not OpenAI's.

## Contents

1. [01_requirement.md](01_requirement.md)
2. [02_solutions/](02_solutions/)

## Data

Bring your own plain-text corpus (a public-domain book from
[Project Gutenberg](https://www.gutenberg.org/) works well — pick
something with a distinctive, consistent style, per Lesson 064's advice).
`02_solutions/data/generate_fallback_corpus.py` (stdlib only) produces a
synthetic fallback corpus if you want to test the pipeline before sourcing
real text.
