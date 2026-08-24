# Project 011 — Text Classifier: Human vs AI-Generated Text (Baseline)

**Builds on lessons:** [055](../../lessons/055_text-preprocessing-tokenization/README.md)–[057](../../lessons/057_word-embeddings/README.md), and Project 003's classical pipeline
**Difficulty:** Intermediate
**Estimated time:** 3–4 hours

## Objective

A very live, real-world classification task: distinguish human-written from
AI-generated text. This is the **classical-NLP baseline** version — Project
012 (after Module 10) upgrades it with a fine-tuned Transformer. The real
point of this project is comparing **two feature representations** on the
same task: sparse TF-IDF (Lesson 056) vs a dense embedding representation
(Lesson 057, via averaged word vectors or an SVD-factorized embedding) —
does moving to dense embeddings actually help here, and can you tell why or
why not?

## Contents

1. [01_requirement.md](01_requirement.md)
2. [02_solutions/](02_solutions/)

## Data

`02_solutions/data/generate_data.py` (stdlib only) produces
`human_vs_ai.csv` — templated text samples designed to mimic real
stylistic differences between human writing (contractions, informal
phrasing, hedging opinions) and AI-generated writing (formal transition
phrases, hedged formality, uniform structure).
