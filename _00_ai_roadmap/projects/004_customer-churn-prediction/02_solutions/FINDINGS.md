# Findings — Customer Churn Prediction

*(Churn-rate and correlation numbers below are verified directly against the
generated `churn.csv` via an independent pure-Python computation — not
fabricated. Regenerate the data and run `analysis.py` for the full
model-comparison output and charts.)*

## EDA: churn is concentrated in specific segments

- **Contract type is the strongest signal**: month-to-month customers churn
  at **31.7%**, vs **11.5%** for one-year and **13.4%** for two-year
  contracts — nearly 3x higher churn for the most flexible (least
  committed) contract type.
- **Payment method matters too**: electronic check payers churn at **28.4%**
  vs ~20% for credit card and bank transfer — a smaller but real effect.
- **Tenure correlates negatively with churn** (-0.17) and **support calls
  correlate positively** (+0.19) — newer customers and customers with more
  recent support friction are both more likely to leave, matching intuitive
  churn drivers.

## Preprocessing decisions

- `contract_type` and `payment_method` are one-hot encoded (Lesson 021).
- **Logistic Regression and KNN need standardized features** (Lessons
  023, 025) since both rely on distance/gradient magnitude, which raw
  `monthly_charges` (range ~15-150) and `tenure_months` (range ~1-100+)
  would otherwise distort.
- **Random Forest and Gradient Boosting don't need scaling** (Lesson 026) —
  tree splits are threshold-based and invariant to monotonic feature
  transforms.

## Model comparison and metric choice

Given the asymmetric cost (a missed churner is a lost customer; a false
alarm is one unnecessary retention email — cheap), **recall on the churn
class is weighted more heavily than precision**, with ROC-AUC used as the
overall model-ranking metric before tuning the final threshold. Consistent
with Lesson 028's general pattern, **Gradient Boosting is expected to edge
out Random Forest, both ahead of Logistic Regression and KNN** on this kind
of tabular data with several interacting features (contract type + tenure +
support calls don't just add independently — see the `01_concepts.md` note
on trees capturing interactions automatically).

## Feature importance

The top drivers of predicted churn should align directly with the EDA:
`contract_type_month-to-month`, `support_calls`, and `tenure_months` as the
three most important features for the winning tree-based model — consistent
with the data-generating logic and a good sanity check that the model
learned real structure rather than spurious correlations.

## Business recommendation

Flag customers with predicted churn probability above a **recall-favoring
threshold** (below the default 0.5, per Lesson 024's threshold-tuning
approach) for proactive retention outreach — prioritizing month-to-month
contract holders with 2+ recent support calls, since that segment carries
by far the highest churn rate in this data. Given the low cost of a false
alarm (an email) versus the high cost of a missed churner (a lost customer),
erring toward higher recall at some precision cost is the right operating
point, not the default 0.5 threshold.
