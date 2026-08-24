# 01 — Questions: ML System Design

Answer each using the 10-step framework from
[Lesson 081](../../lessons/081_ml-system-design-interview/01_concepts.md),
in writing, time-boxed to 45 minutes per question. Don't peek at
`02_solutions/` until you've written your own answer — the value here is
in noticing which step you skip under time pressure, not in reading a
model answer cold.

1. **Design a real-time fraud detection system for credit card
   transactions.** Transactions must be approved or declined in under
   200ms. Address the label-delay problem explicitly: confirmed fraud
   labels (chargebacks) often arrive weeks after the transaction.

2. **Design a search ranking system for a job-listings site** (rank
   job postings for a given search query + user profile). Address how
   you'd evaluate ranking quality specifically (not just classification
   accuracy — ranking needs its own metrics).

3. **Design a "what to watch next" video recommendation system** shown
   after a user finishes a video. Distinguish this explicitly from a
   homepage recommendation system (Lesson 081, Q2) — the available
   context (the just-watched video) is much richer and more immediate
   here.

4. **Design an autocomplete/typeahead system for a search bar**, ranking
   suggested completions as the user types. Latency budget: under 50ms
   per keystroke. Discuss what's cached/precomputed vs. computed live.

5. **Design an anomaly detection system for server infrastructure
   metrics** (CPU, memory, request latency, error rate across thousands
   of hosts) that pages an on-call engineer. Address the alert-fatigue
   problem explicitly: a system with too many false positives gets
   ignored, regardless of its recall.
