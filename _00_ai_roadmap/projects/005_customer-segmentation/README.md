# Project 005 — Customer Segmentation

**Builds on lessons:** [031](../../lessons/031_pca/README.md)–[034](../../lessons/034_dbscan/README.md) (all of Module 5)
**Difficulty:** Intermediate
**Estimated time:** 3–4 hours

## Objective

A classic real-world unsupervised task: given customer purchase behavior
with **no labels at all**, find meaningful segments a marketing team could
actually act on (e.g. "high value loyal," "at-risk," "new/low engagement").
This is the Module 5 capstone — PCA for visualization, K-Means for the main
segmentation, DBSCAN/hierarchical for a sanity-check second opinion.

## Contents

1. [01_requirement.md](01_requirement.md)
2. [02_solutions/](02_solutions/)

## Data

`02_solutions/data/generate_data.py` (stdlib only) produces `customers.csv`
— RFM-style behavioral data (Recency, Frequency, Monetary) for 800
customers, generated from 4 latent segments the clustering should
rediscover.
