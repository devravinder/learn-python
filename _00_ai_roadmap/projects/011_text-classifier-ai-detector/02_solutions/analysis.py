"""Reference solution: TF-IDF vs SVD-factorized dense embeddings for
human-vs-AI text classification.

Run:
    python data/generate_data.py
    python analysis.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

DATA_PATH = Path(__file__).parent / "data" / "human_vs_ai.csv"


def eda(df):
    df["n_words"] = df["text"].str.split().apply(len)
    df["unique_ratio"] = df["text"].apply(lambda t: len(set(t.lower().split())) / len(t.split()))
    print("=== Avg words per text, by label ===")
    print(df.groupby("label")["n_words"].mean())
    print("\n=== Avg unique-word ratio, by label ===")
    print(df.groupby("label")["unique_ratio"].mean())

    hallmark_phrases = ["in conclusion", "furthermore", "it is important to note"]
    for phrase in hallmark_phrases:
        rate = df["text"].str.lower().str.contains(phrase).groupby(df["label"]).mean()
        print(f"\n'{phrase}' occurrence rate by label:\n{rate}")


def evaluate_pipeline(name, vectorizer_pipeline, X_train, X_test, y_train, y_test):
    X_train_feat = vectorizer_pipeline.fit_transform(X_train)
    X_test_feat = vectorizer_pipeline.transform(X_test)

    results = {}
    for clf_name, clf in [("LogisticRegression", LogisticRegression(max_iter=1000)), ("NaiveBayes", MultinomialNB())]:
        try:
            clf.fit(X_train_feat, y_train)
            preds = clf.predict(X_test_feat)
            print(f"\n--- {name} + {clf_name} ---")
            print(classification_report(y_test, preds))
            results[clf_name] = clf
        except ValueError as e:
            print(f"\n--- {name} + {clf_name}: skipped ({e}) ---")  # e.g. NB needs non-negative features
    return results, X_train_feat, X_test_feat


def main():
    df = pd.read_csv(DATA_PATH)
    eda(df)

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, stratify=df["label"], random_state=0
    )

    print("\n" + "=" * 20 + " TF-IDF (sparse) " + "=" * 20)
    tfidf = TfidfVectorizer()
    evaluate_pipeline("TF-IDF", tfidf, X_train, X_test, y_train, y_test)

    print("\n" + "=" * 20 + " TF-IDF + SVD (dense embedding, 50 dims) " + "=" * 20)
    svd_pipeline = make_pipeline(TfidfVectorizer(), TruncatedSVD(n_components=50, random_state=0))
    # NaiveBayes needs non-negative features - SVD output can be negative, so only LogisticRegression applies here
    X_train_svd = svd_pipeline.fit_transform(X_train)
    X_test_svd = svd_pipeline.transform(X_test)
    clf = LogisticRegression(max_iter=1000).fit(X_train_svd, y_train)
    print(classification_report(y_test, clf.predict(X_test_svd)))

    # generalization stress test
    print("\n=== Generalization stress test (novel, hand-written sentences) ===")
    novel_texts = [
        "honestly not sure how i feel about all this ai stuff, kinda weird ngl",
        "In summary, this topic warrants further investigation and analysis by researchers.",
        "The weather today is nice.",
        "This is a great product I really enjoyed using it every day.",
    ]
    tfidf_full = TfidfVectorizer().fit(df["text"])
    X_full = tfidf_full.transform(df["text"])
    clf_full = LogisticRegression(max_iter=1000).fit(X_full, df["label"])
    for text in novel_texts:
        pred = clf_full.predict(tfidf_full.transform([text]))[0]
        print(f"{text!r} -> predicted {'AI' if pred == 1 else 'human'}")


if __name__ == "__main__":
    main()
