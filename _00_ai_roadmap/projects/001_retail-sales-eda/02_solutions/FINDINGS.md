# Findings — Retail Sales EDA

*(Reference write-up. Numbers below are from the actual generated dataset —
regenerate with `data/generate_data.py` and run `analysis.py` to reproduce them
and the charts in `charts/`.)*

## Data quality

Out of 3,030 exported rows: 45 had a missing quantity (recovered from
`total_amount / unit_price`), 30 had a missing city (dropped — can't attribute
revenue to an unknown location), 30 were exact duplicates of another row
(dropped), and 15 had a negative total amount, almost certainly refunds
mis-recorded as sales (excluded from revenue totals). After cleaning: 2,955
usable rows.

## Sales trend

Revenue is heavily seasonal: November and December are the two strongest
months in both years (Nov 2024: ~$55.8K, Dec 2024: ~$52.5K, Nov 2025: ~$61.0K,
Dec 2025: ~$84.3K), roughly 3–5x a typical mid-year month (e.g. June 2024:
~$8.2K). Year-over-year, the Nov/Dec peak grew rather than shrank, and Feb 2025
also stands out as unusually strong (~$26.5K) relative to its neighbors — worth
a follow-up question to the business (promotion? one large bulk order?) rather
than assuming it's noise.

## Category & city breakdown

**Electronics dominates revenue** (~$387.8K of the cleaned total), far ahead of
Clothing (~$76.9K), Home (~$64.5K), Toys (~$24.0K), and Grocery (~$19.2K) —
expected, since electronics (laptops especially) carry a much higher unit
price, not necessarily more units sold. Revenue is close to evenly split
across cities (NY ~$149.2K, CHI ~$144.3K, SF ~$139.7K, LA ~$139.2K) — no single
city is an outlier, so this dataset doesn't support a "focus on city X" story.

## Order value distribution

Mean order value (~$193.71) is roughly **4x the median** (~$47.82), with a max
of ~$4,996 — a strong right skew driven by a relatively small number of
high-ticket electronics orders (laptops). This is exactly the pattern where
reporting "average order value" alone would be misleading; median (or
reporting electronics separately) gives a truer picture of a typical
transaction.

## Bottom line for stakeholders

Sales are strongly seasonal (holiday-driven) and revenue-concentrated in
Electronics due to price, not volume. Data quality is decent (~2.5% of rows
needed cleaning) but the export pipeline should be checked for the source of
duplicate rows and refund-as-negative-sale records before this feed is trusted
for automated reporting.
