"""Reference solution: classical text classification with TF-IDF +
Naive Bayes / Logistic Regression.

Run:
    python data/generate_data.py
    python analysis.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

DATA_DIR = Path(__file__).parent / "data"


def run_pipeline(csv_name, label):
    print(f"\n{'='*20} {label} {'='*20}")
    df = pd.read_csv(DATA_DIR / csv_name)

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, stratify=df["label"], random_state=0
    )

    # TF-IDF over CountVectorizer: down-weights very common words across the
    # whole corpus (e.g. "the", "a") automatically, usually a stronger
    # default for text classification than raw counts.
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    nb = MultinomialNB().fit(X_train_vec, y_train)
    lr = LogisticRegression(max_iter=1000).fit(X_train_vec, y_train)

    print("--- Naive Bayes ---")
    print(classification_report(y_test, nb.predict(X_test_vec)))
    print("--- Logistic Regression ---")
    print(classification_report(y_test, lr.predict(X_test_vec)))

    return vectorizer, nb, lr, X_test, y_test, X_test_vec


def threshold_for_precision(model, X_vec, y_true, target_precision=0.98):
    probs = model.predict_proba(X_vec)[:, 1]
    for t in np.linspace(0.5, 0.99, 50):
        preds = (probs >= t).astype(int)
        p = precision_score(y_true, preds, zero_division=0)
        if p >= target_precision:
            r = recall_score(y_true, preds, zero_division=0)
            return t, p, r
    return None, None, None


def error_analysis(model, vectorizer, X_test, y_test, X_test_vec, n=5):
    preds = model.predict(X_test_vec)
    wrong_idx = np.where(preds != y_test.to_numpy())[0]
    print(f"\n{len(wrong_idx)} misclassified out of {len(y_test)}")
    for i in wrong_idx[:n]:
        print(f"  text: {X_test.iloc[i]!r} | true={y_test.iloc[i]} pred={preds[i]}")


def top_spam_words(vectorizer, nb_model, n=15):
    feature_names = np.array(vectorizer.get_feature_names_out())
    log_prob_diff = nb_model.feature_log_prob_[1] - nb_model.feature_log_prob_[0]
    top_idx = np.argsort(log_prob_diff)[-n:][::-1]
    return list(feature_names[top_idx])


def main():
    vec1, nb1, lr1, Xte1, yte1, Xte1_vec = run_pipeline("sms_spam.csv", "SMS Spam")

    t, p, r = threshold_for_precision(nb1, Xte1_vec, yte1, target_precision=0.98)
    print(f"\nThreshold for >=98% precision: {t}, achieved precision={p}, recall={r}")

    error_analysis(nb1, vec1, Xte1, yte1, Xte1_vec)

    print("\nTop spam-indicative words:", top_spam_words(vec1, nb1))

    run_pipeline("fake_reviews.csv", "Fake Reviews")


if __name__ == "__main__":
    main()
