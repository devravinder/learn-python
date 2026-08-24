"""Reference solution: end-to-end churn prediction, comparing 4 model
families from Module 4.

Run:
    python data/generate_data.py
    python analysis.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path(__file__).parent / "data" / "churn.csv"
CHARTS_DIR = Path(__file__).parent / "charts"
CHARTS_DIR.mkdir(exist_ok=True)


def eda(df):
    print("=== Churn rate by contract type ===")
    print(df.groupby("contract_type")["churned"].mean())
    print("\n=== Churn rate by payment method ===")
    print(df.groupby("payment_method")["churned"].mean())

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    df.groupby("contract_type")["churned"].mean().plot(kind="bar", ax=axes[0], title="Churn by contract")
    df.groupby("payment_method")["churned"].mean().plot(kind="bar", ax=axes[1], title="Churn by payment")
    sns.histplot(data=df, x="tenure_months", hue="churned", ax=axes[2], bins=30, kde=True)
    axes[2].set_title("Tenure distribution by churn")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "eda.png")
    plt.close(fig)


def prepare_features(df):
    df_encoded = pd.get_dummies(df, columns=["contract_type", "payment_method"], drop_first=True)
    feature_cols = [c for c in df_encoded.columns if c not in ("customer_id", "churned")]
    X = df_encoded[feature_cols]
    y = df_encoded["churned"]
    return X, y, feature_cols


def main():
    df = pd.read_csv(DATA_PATH)
    eda(df)

    X, y, feature_cols = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=0
    )

    # KNN and Logistic Regression need scaled features (distance/gradient
    # based); tree-based models (RF, GB) are scale-invariant (Lesson 026).
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "LogisticRegression": (LogisticRegression(max_iter=1000), True),
        "KNN": (KNeighborsClassifier(n_neighbors=15), True),
        "RandomForest": (RandomForestClassifier(n_estimators=300, random_state=0), False),
        "GradientBoosting": (GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, random_state=0), False),
    }

    results = {}
    for name, (model, needs_scaling) in models.items():
        Xtr = X_train_scaled if needs_scaling else X_train
        Xte = X_test_scaled if needs_scaling else X_test
        model.fit(Xtr, y_train)
        probs = model.predict_proba(Xte)[:, 1]
        auc = roc_auc_score(y_test, probs)
        results[name] = (model, probs, auc)
        print(f"\n=== {name} (AUC={auc:.3f}) ===")
        print(classification_report(y_test, model.predict(Xte)))

    # Primary metric: recall-weighted (catching churners matters more than a
    # false alarm, since a missed churner is a lost customer vs one extra
    # retention email) -> pick highest AUC as the general-purpose ranking,
    # then tune threshold for recall on the winner.
    best_name = max(results, key=lambda k: results[k][2])
    print(f"\nBest model by AUC: {best_name}")

    plt.figure()
    for name, (_, probs, auc) in results.items():
        fpr, tpr, _ = roc_curve(y_test, probs)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.2f})")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.legend()
    plt.savefig(CHARTS_DIR / "roc_comparison.png")
    plt.close()

    # feature importance for the winning tree-based model (or coefficients if linear)
    best_model = results[best_name][0]
    if hasattr(best_model, "feature_importances_"):
        importances = pd.Series(best_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    else:
        importances = pd.Series(best_model.coef_[0], index=feature_cols).sort_values(key=abs, ascending=False)
    print(f"\n=== {best_name} feature importance/coefficients ===")
    print(importances)


if __name__ == "__main__":
    main()
