# 01 — Requirement: Customer Segmentation

## The brief

> "Marketing wants to run different campaigns for different types of
> customers but we've never actually grouped them systematically. Find
> natural customer segments in our purchase data."

## Dataset schema (`customers.csv`)

| Column | Type | Notes |
|---|---|---|
| `customer_id` | int | |
| `recency_days` | int | days since last purchase (lower = more recent) |
| `frequency` | int | number of purchases in the last year |
| `monetary` | float | total spend in the last year |
| `avg_basket_size` | float | average items per order |

There is **no label column** — this is unsupervised, by design.

## What to produce

1. **EDA**: distributions of each feature, pairwise scatter plots
   (`seaborn.pairplot` or manual). Do you see any visual hints of natural
   groupings before running any algorithm?

2. **Preprocessing**: standardize features (Lesson 008/031 — required before
   K-Means/DBSCAN since they're distance-based).

3. **Dimensionality reduction for visualization**: reduce to 2D with PCA
   (Lesson 031) so you can visually inspect clustering results throughout.

4. **K-Means segmentation**: use the elbow method and silhouette score
   (Lesson 032) to choose `k`. Fit the final model and visualize segments in
   the 2D PCA space.

5. **Second opinion**: fit Hierarchical clustering (Lesson 033) with the
   same `k` and DBSCAN (Lesson 034, tune `eps` via a k-distance plot). Do
   all three methods roughly agree on the segment structure? Where do they
   disagree, and why (relate to each algorithm's assumptions)?

6. **Profile each segment**: for your final K-Means segments, compute the
   mean of each original feature (recency/frequency/monetary/basket size)
   per segment. Give each segment a business-friendly name (e.g. "Champions,"
   "At Risk," "New Customers," "Occasional Buyers") based on its profile.

7. **Business recommendation**: for each named segment, write 1-2 sentences
   on what marketing action makes sense (e.g. a win-back campaign for an
   "At Risk" segment with high recency and low frequency).

## Constraints

- Genuinely unsupervised — do not use any hidden "true segment" information
  to pick `k`; justify your choice using only the internal metrics from
  Module 5.
- Don't peek at `02_solutions/` before producing your own segment profiles.
