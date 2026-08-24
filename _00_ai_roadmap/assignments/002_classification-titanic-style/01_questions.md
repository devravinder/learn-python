# 01 — Questions

Dataset: `02_solutions/data/titanic_synthetic.csv` (regenerate via
`generate_data.py` if needed). Columns: `passenger_id`, `pclass` (1/2/3),
`sex`, `age` (has missing values), `fare`, `sibsp` (siblings/spouses aboard),
`survived` (target, 0/1).

1. Load the data, report missing values per column, and impute missing `age`
   with the **median age per `pclass`** (not the overall median — reasoning:
   age distributions likely differ by class; justify in a comment).

2. One-hot encode `sex` (drop first). Split into train/test (80/20,
   `stratify` on `survived`).

3. Fit a `LogisticRegression` using `pclass`, `sex_male`, `age`, `fare`,
   `sibsp` as features. Report `classification_report` and confusion matrix
   on the test set.

4. Report the model's coefficients. Does `sex` have the effect on survival
   you'd expect, given how the data was generated (favor looking at the
   generator's logic if you want to check your interpretation, but try to
   answer from the coefficients alone first)?

5. Compute ROC-AUC on the test set. Plot the ROC curve.

6. Fit a second model using **only** `pclass` and `fare` (dropping `sex`,
   `age`, `sibsp`). Compare its ROC-AUC to the full model's — how much
   predictive power comes specifically from `sex` and `age`?

7. Find the classification threshold that maximizes F1 on the test set, and
   report precision/recall at that threshold vs the default 0.5.

8. Write 3–4 sentences summarizing what predicts survival in this dataset,
   as if reporting to someone who hasn't seen the data — plain language, no
   code.
