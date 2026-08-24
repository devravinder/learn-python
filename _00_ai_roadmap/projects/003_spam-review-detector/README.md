# Project 003 — Spam & Fake Review Detector

**Builds on lessons:** [023](../../lessons/023_logistic-regression/README.md), [024](../../lessons/024_classification-metrics/README.md), [030](../../lessons/030_naive-bayes/README.md)
**Difficulty:** Intermediate
**Estimated time:** 3–4 hours

## Objective

Your first "predict a label from text" project — classical NLP, no deep
learning yet. Build and compare two classifiers (Naive Bayes and Logistic
Regression) on TF-IDF features to detect spam messages, then reuse the exact
same pipeline on a second, harder task (fake product reviews) to see how
well the approach generalizes to a different text-classification problem.

## Contents

1. [01_requirement.md](01_requirement.md)
2. [02_solutions/](02_solutions/)

## Data

`02_solutions/data/generate_data.py` (stdlib only) produces two datasets:
`sms_spam.csv` and `fake_reviews.csv`.
