# Project 007 — Time-Series Forecasting with RNN/LSTM

**Builds on lessons:** [045](../../lessons/045_rnn-fundamentals/README.md)–[047](../../lessons/047_seq2seq-encoder-decoder/README.md) (Module 7 capstone)
**Difficulty:** Intermediate
**Estimated time:** 3–4 hours

## Objective

Apply LSTMs to a genuinely sequential real-world task: forecasting future
values from a time series with trend, weekly seasonality, and noise. The
twist that makes this a *real* exercise, not just an LSTM demo: you must
also build simple statistical baselines and prove the LSTM actually beats
them — a habit worth having for any forecasting task, since simple
baselines are stronger than people expect.

## Contents

1. [01_requirement.md](01_requirement.md)
2. [02_solutions/](02_solutions/)

## Data

`02_solutions/data/generate_data.py` (stdlib only) produces `sales.csv` —
2 years of daily synthetic sales data with trend, weekly seasonality, and
noise.
