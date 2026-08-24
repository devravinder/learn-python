# Findings — Time-Series Forecasting

*(Baseline numbers below are computed directly from the actual generated
`sales.csv` via plain Python arithmetic — verified, not fabricated. The
LSTM numbers are **not** independently verified here — no PyTorch in the
authoring sandbox, unlike the baselines — so they're presented as expected
qualitative behavior. Run `analysis.py` yourself for real LSTM numbers.)*

## Baseline results (verified against the real generated data)

| Baseline | MAE | RMSE |
|---|---|---|
| Naive (today = yesterday) | 14.23 | 18.16 |
| **Seasonal naive (today = same day last week)** | **7.89** | **10.39** |
| Moving average (last 7 days) | 12.79 | 15.23 |

**Seasonal naive is by far the strongest baseline here** — nearly 2x better
than plain naive and clearly ahead of the moving average. This makes sense
given the data: sales have strong, consistent weekly seasonality (weekends
run higher), so "same day last week" captures most of the predictable
signal directly, while a same-day predictor (naive) or a smoothed
7-day-average predictor (moving average) both wash out that weekly pattern.

## What this means for evaluating the LSTM

**The LSTM must beat a 7.89 MAE / 10.39 RMSE bar to be worth using at all**
on this data — not the weaker naive/moving-average baselines. This is
exactly the kind of check the project brief asks you not to skip: on a
series this cleanly seasonal, a much simpler seasonal-naive rule is a
legitimately strong competitor, and it would be easy (and wrong) to declare
victory for the LSTM by only comparing against the weakest baseline.

## Expected LSTM behavior

Given the series is dominated by a simple, consistent weekly pattern plus a
mild linear trend, expect the LSTM to **learn the weekly seasonality
reasonably well** (it has direct access to the last 14 days, more than
enough to see the pattern) and land **in the same ballpark as seasonal
naive**, possibly edging slightly ahead by also picking up the linear trend
that seasonal-naive ignores — but don't expect a dramatic win. On genuinely
complex, less-regular real-world series (multiple overlapping seasonalities,
regime changes, external events), the gap between a naive seasonal baseline
and a learned model tends to be much larger — this dataset was deliberately
kept simple enough to train quickly, which also means it's a favorable case
for simple baselines.

## Multi-step forecasting

Expect mean absolute error to **increase with forecast horizon** (1-day-
ahead easier than 7-days-ahead) — each autoregressive step feeds the
previous step's prediction (with its own error) back in as input, so
errors compound over the horizon (Lesson 047's exposure-bias point, applied
directly). This is a universal property of autoregressive multi-step
forecasting, not specific to this dataset or model.
