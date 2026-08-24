# 01 — Concepts: Bag-of-Words & TF-IDF

## Bag-of-words: a document as a vector of word counts

Represent each document as a vector over the entire vocabulary, where
each entry is how many times that word appears — completely discarding
word order (hence "bag," not "sequence," of words).

```python
from sklearn.feature_extraction.text import CountVectorizer

docs = ["the cat sat on the mat", "the dog sat on the log"]
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(docs)
print(vectorizer.get_feature_names_out())
print(X.toarray())
```

This is exactly the representation Project 003's `MultinomialNB` consumed
(Lesson 030) — each document becomes a point in a very high-dimensional,
very sparse space (vocabulary size = dimensions, almost all zero for any
single short document).

## The problem with raw counts: common words dominate

Words like "the," "is," "and" appear in almost every document and carry
little information about what makes a *specific* document distinctive, but
raw counts weight them heavily just because they're frequent.

## TF-IDF: down-weighting words that appear everywhere

**Term Frequency-Inverse Document Frequency** combines two signals:

```
TF(t, d) = count of term t in document d  (optionally normalized by doc length)
IDF(t) = log(N / (1 + document_count(t)))     N = total number of documents

TF-IDF(t, d) = TF(t, d) * IDF(t)
```

- **High TF-IDF**: a word that appears often in *this* document but rarely
  across the *whole corpus* — a genuinely distinctive term for this
  document.
- **Low TF-IDF**: either rare in this document, or common across nearly
  every document (like "the," whose `IDF` approaches 0 since
  `document_count(t) ≈ N`).

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(docs)
```

This is exactly why Project 003 used `TfidfVectorizer` over
`CountVectorizer` for the final pipeline — down-weighting universally
common words tends to help classifiers focus on genuinely distinguishing
vocabulary.

## Cosine similarity between documents (revisiting Lesson 010)

Once documents are vectors, Lesson 010's cosine similarity directly applies
to measure how similar two documents are:

```python
from sklearn.metrics.pairwise import cosine_similarity
similarity_matrix = cosine_similarity(X)
```

This is the basis of classical document search/retrieval: represent a
query the same way as documents, then rank documents by cosine similarity
to the query vector — a simplified ancestor of Lesson 069's
Retrieval-Augmented Generation, which does the same *kind* of nearest-
neighbor lookup but with dense embedding vectors (Lesson 057) instead of
sparse TF-IDF vectors.

## N-grams: a partial fix for "no word order"

A **bigram** ("2-gram") is a pair of adjacent words treated as a single
unit — `"not good"` becomes its own feature, distinct from `"not"` and
`"good"` separately, partially recovering some local word-order
information lost by pure bag-of-words.

```python
vectorizer = CountVectorizer(ngram_range=(1, 2))   # unigrams AND bigrams
```

This helps somewhat (e.g. distinguishing "not good" from "good") but
doesn't scale — the vocabulary grows very fast with n-gram length, and
long-range word order (beyond a few adjacent words) is still completely
lost. This limitation is *exactly* the motivation for Lesson 057's
embeddings and, ultimately, attention (Lesson 058) — which captures
relationships between *any* two words in a sentence, regardless of
distance, without an exploding feature count.

## What bag-of-words/TF-IDF are still good for

Despite the limitations, TF-IDF remains a strong, fast, interpretable
baseline for tasks like document classification, search/retrieval ranking,
and keyword extraction — Project 003's classical spam/fake-review detector
performed strongly with exactly this representation. The lesson isn't
"never use this," it's "know precisely what it throws away," so you can
recognize when a task (like distinguishing subtly different meanings, or
generating text) genuinely needs the richer representations from Lesson
057 onward.
