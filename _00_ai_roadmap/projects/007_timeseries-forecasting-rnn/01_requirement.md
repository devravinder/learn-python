# 01 — Requirement: Time-Series Forecasting

## The brief

> "Forecast next week's daily sales from historical data so we can plan
> inventory."

## Dataset

`sales.csv`: `date`, `sales` — 730 days (2 years) of daily sales with an
upward trend, strong weekly seasonality (weekends higher), and noise.

## What to produce

1. **EDA**: plot the full series. Identify trend and weekly seasonality
   visually (e.g. plot average sales by day-of-week).

2. **Train/test split respecting time order**: the **last 60 days** are the
   test set; everything before is training data. (Never randomly shuffle a
   time series split — Lesson 024's `stratify` doesn't apply here; order
   matters and the model must never see future data during training.)

3. **Baselines** (build these *before* the LSTM, and don't skip them):
   - **Naive/persistence**: predict tomorrow = today.
   - **Seasonal naive**: predict this Monday = last Monday (i.e. lag-7).
   - **Moving average**: predict tomorrow = average of the last 7 days.

   Report MAE and RMSE (Lesson 018) for each baseline on the test set.

4. **LSTM forecaster**: build a sliding-window dataset (e.g. use the last
   14 days to predict the next 1 day), normalize the series (Lesson 008 —
   critical for LSTM training stability), and train an LSTM-based model
   (Lesson 046). Report MAE/RMSE on the test set in the *original* (
   un-normalized) scale — don't report normalized-scale error, it's not
   interpretable.

5. **Compare**: does the LSTM actually beat all three baselines? If it
   doesn't beat the seasonal-naive or moving-average baseline, that's a
   real, valid, reportable finding — not a failure to hide. Report which
   won and by how much.

6. **Multi-step forecasting**: extend the LSTM to forecast 7 days ahead
   autoregressively (predict day 1, feed it back in to help predict day 2,
   etc. — Lesson 047's autoregressive generation pattern). Does forecast
   error grow as you predict further into the future? Plot error vs
   forecast horizon (1 day ahead, 2 days ahead, ..., 7 days ahead).

## Constraints

- Respect time order everywhere — no shuffled train/test splits, no
  normalizing using statistics computed from the test period.
- Don't peek at `02_solutions/` before you have your own baseline
  comparison table.
