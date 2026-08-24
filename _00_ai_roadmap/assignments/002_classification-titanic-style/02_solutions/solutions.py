"""Reference solutions for Assignment 002."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    precision_score, recall_score, f1_score,
)
from sklearn.model_selection import train_test_split

DATA_PATH = Path(__file__).parent / "data" / "titanic_synthetic.csv"

df = pd.read_csv(DATA_PATH)

# --- Q1: missing values + per-class median imputation ---
print("Q1 missing values:\n", df.isna().sum(), "\n")
# Median per pclass, not overall: wealthier/1st-class passengers likely skew
# older on average than 3rd class, so a single overall median would bias
# imputed ages differently across classes.
df["age"] = df.groupby("pclass")["age"].transform(lambda s: s.fillna(s.median()))

# --- Q2: encode + split ---
df_encoded = pd.get_dummies(df, columns=["sex"], drop_first=True)
features = ["pclass", "sex_male", "age", "fare", "sibsp"]
X = df_encoded[features]
y = df_encoded["survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=0
)

# --- Q3: fit + report ---
model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
preds = model.predict(X_test)
print("Q3 classification report:\n", classification_report(y_test, preds))
print("Q3 confusion matrix:\n", confusion_matrix(y_test, preds), "\n")

# --- Q4: coefficients ---
print("Q4 coefficients:", dict(zip(features, model.coef_[0].round(3))), "\n")
# Expected: sex_male strongly negative (being male reduces survival log-odds
# sharply), matching the generator's `+1.8 if female` term.

# --- Q5: ROC-AUC ---
probs = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, probs)
print("Q5 ROC-AUC:", auc)
fpr, tpr, _ = roc_curve(y_test, probs)
plt.plot(fpr, tpr, label=f"AUC={auc:.2f}")
plt.plot([0, 1], [0, 1], "--", color="gray")
plt.legend()
plt.savefig(Path(__file__).parent / "q5_roc_curve.png")
plt.close()

# --- Q6: pclass + fare only ---
X_reduced = df_encoded[["pclass", "fare"]]
Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    X_reduced, y, test_size=0.2, stratify=y, random_state=0
)
model_reduced = LogisticRegression(max_iter=1000).fit(Xr_train, yr_train)
probs_reduced = model_reduced.predict_proba(Xr_test)[:, 1]
auc_reduced = roc_auc_score(yr_test, probs_reduced)
print("Q6 full-model AUC:", auc, "reduced-model (pclass+fare only) AUC:", auc_reduced, "\n")
# Expected: reduced model's AUC noticeably lower, since sex/age carry real
# independent signal per the data-generating process.

# --- Q7: best-F1 threshold ---
import numpy as np
thresholds = np.linspace(0.1, 0.9, 17)
best_f1, best_t = -1, 0.5
for t in thresholds:
    p = (probs >= t).astype(int)
    f1 = f1_score(y_test, p, zero_division=0)
    if f1 > best_f1:
        best_f1, best_t = f1, t

preds_05 = (probs >= 0.5).astype(int)
preds_best = (probs >= best_t).astype(int)
print("Q7 threshold 0.5:", precision_score(y_test, preds_05), recall_score(y_test, preds_05))
print(f"Q7 threshold {best_t}:", precision_score(y_test, preds_best), recall_score(y_test, preds_best), "F1:", best_f1)
