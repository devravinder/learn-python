# 02 — Practicals: Bag-of-Words & TF-IDF

```python
docs = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "cats and dogs are great pets",
]
```

1. Build a bag-of-words representation with `CountVectorizer` and print the
   resulting matrix with column labels (`get_feature_names_out()`). Which
   words have the highest raw count across the corpus?

2. Implement TF-IDF **from scratch** (no `sklearn`) using the formula from
   `01_concepts.md`: compute document frequency for every word, then TF-IDF
   for each document. Confirm that "the," "sat," and "on" (which appear in
   documents 0 and 1, 2 out of 3 documents) get an IDF of **exactly 0**
   with this formula and this corpus — work out why by hand from the IDF
   formula before checking in code.

3. Compare your from-scratch TF-IDF to `sklearn.feature_extraction.text.TfidfVectorizer`
   on the same docs. The values won't match exactly (sklearn applies extra
   normalization by default — check its docs for `norm` and `smooth_idf`)
   but the *ranking* of important words per document should agree.

4. Compute cosine similarity (Lesson 010) between all pairs of the 3
   documents using your TF-IDF vectors. Which two documents are most
   similar? Does that match your intuition about their content?

5. Rebuild the vectorizer with `ngram_range=(1,2)` (unigrams + bigrams).
   How much does the vocabulary size grow? Find a bigram feature that
   captures something a unigram-only representation would miss.

6. Using Project 003's `sms_spam.csv` (regenerate if needed), compare
   classification accuracy (Naive Bayes, Lesson 030) using `CountVectorizer`
   vs `TfidfVectorizer` features. Does TF-IDF's down-weighting of common
   words help, hurt, or make little difference on this dataset?
