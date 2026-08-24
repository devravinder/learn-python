# 01 — Requirement: Customer Churn Prediction

## The brief

> "We're losing customers and don't know who's at risk until they've
> already left. Build a model to flag customers likely to churn next month
> so the retention team can reach out proactively."

## Dataset schema (`churn.csv`)

| Column | Type | Notes |
|---|---|---|
| `customer_id` | int | |
| `tenure_months` | int | how long they've been a customer |
| `contract_type` | str | `month-to-month`, `one-year`, `two-year` |
| `monthly_charges` | float | |
| `total_charges` | float | |
| `support_calls` | int | number of support calls in the last 3 months |
| `payment_method` | str | `credit_card`, `bank_transfer`, `electronic_check` |
| `is_senior` | int | 0/1 |
| `has_dependents` | int | 0/1 |
| `churned` | int | target, 0/1, ~20% positive |

## What to produce

1. **EDA**: distributions of numeric features, churn rate by
   `contract_type` and `payment_method`, correlation of numeric features
   with churn. At least 3 charts.

2. **Preprocessing pipeline**: handle categorical encoding, feature scaling
   where needed (which models from Module 4 need it and which don't —
   answer explicitly), and a train/test split stratified on `churned`.

3. **Model comparison**: train and evaluate at least 4 models spanning what
   you learned in Module 4 — e.g. Logistic Regression (Lesson 023), Random
   Forest (Lesson 027), Gradient Boosting (Lesson 028), and one more of your
   choice (KNN, SVM, or Naive Bayes). Use consistent preprocessing across
   all of them (reuse the same train/test split).

4. **Metric choice, justified**: churn prediction is imbalanced (Lesson
   018/024) and has asymmetric costs (missing a churner costs a lost
   customer; a false alarm costs one retention email). Pick a primary
   metric and threshold, and justify the choice in writing.

5. **Model selection**: pick a final model based on your chosen metric on
   the test set (not training performance). Report a full
   `classification_report` and ROC-AUC for the winner.

6. **Feature importance / interpretation**: for your final model, report
   what drives churn (coefficients if linear, `feature_importances_` if
   tree-based). Do the top drivers make business sense?

7. **Business recommendation**: 3-5 sentences translating the model into
   action — e.g. "flag customers with predicted churn probability above X;
   this catches Y% of actual churners at a cost of Z% false-alarm rate,
   which the retention team can handle."

## Constraints

- No deep learning yet (that's Module 6+) — Module 4/5 techniques only.
- Don't peek at `02_solutions/` before you have your own comparison table.
