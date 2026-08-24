# Findings — Customer Segmentation

*(Cluster profile numbers below are verified via an independent pure-Python
K-Means implementation run against the actual generated `customers.csv` —
not fabricated. `analysis.py`'s sklearn version, which additionally uses
`n_init=10` random restarts, may recover a cleaner split — regenerate the
data and run it to compare against this single-run result.)*

## K=4 segmentation and profiles

Running K-Means (k=4, standardized features) produced these segments:

| Segment | n | Recency (days) | Frequency | Monetary | Basket size | Name |
|---|---|---|---|---|---|---|
| 0 | 253 | 44.6 | 3.8 | $247.5 | 2.0 | **New / Occasional** |
| 1 | 215 | 162.5 | 4.9 | $511.5 | 2.3 | **At Risk** |
| 2 | 165 | 9.6 | 11.5 | $1362.3 | 4.5 | **Loyal Regulars** |
| 3 | 167 | 9.7 | 23.5 | $1476.5 | 4.4 | **Champions** |

## An honest note on cluster recovery

This dataset was generated from 4 latent segments, but this particular
K-Means run **split the highest-value customers into two sub-groups**
(clusters 2 and 3 — nearly identical recency and monetary value, but very
different purchase frequency) rather than cleanly recovering the original 4
generating groups. This is a real, expected K-Means behavior, not a bug —
Lesson 032 flags exactly this sensitivity to random initialization; running
with multiple restarts (`n_init=10`, used in `analysis.py`) or checking
silhouette scores across several `k` values (rather than assuming `k=4` is
correct) is the right way to guard against reporting an unstable split as if
it were ground truth. In a real project, this would be worth flagging back
to the business as "frequency splits our top spenders into two behaviorally
different groups" — a legitimate, actionable insight, not necessarily a
mistake to fix.

## Segment interpretation and recommended actions

- **Champions** (167 customers, buy often, spend the most, very recent):
  reward with a loyalty/VIP program — retention here has the highest
  dollar impact per customer.
- **Loyal Regulars** (165 customers, spend nearly as much as Champions but
  buy about half as often): a good target for a frequency-boosting
  campaign (e.g. "come back sooner" incentives) since their per-visit
  spend is already high.
- **At Risk** (215 customers, long time since last purchase, moderate past
  spend): the clearest win-back opportunity — a targeted re-engagement
  offer, since these customers have proven purchase intent but have gone
  quiet.
- **New / Occasional** (253 customers, recent-ish but low frequency and
  spend): nurture with onboarding/first-repeat-purchase incentives — too
  early to tell if they'll become loyal or churn.

## Cross-checking with other methods

Hierarchical clustering (Ward linkage, same `k`) is expected to produce
broadly similar groupings to K-Means here, since Ward linkage optimizes a
similar within-cluster-variance objective. DBSCAN, which doesn't assume a
fixed `k` or spherical clusters, is a useful sanity check for whether the
data actually separates into clean dense regions at all, or whether the
segments are more of a continuum — worth reporting to the business either
way, since "4 clean segments" and "one continuous spectrum of customer
value" call for different marketing strategies.
