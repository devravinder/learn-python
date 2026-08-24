# 03 — Solutions: Bag-of-Words & TF-IDF

*(This code was actually run to produce the numbers below — including a
genuinely interesting, non-obvious result in Q4, kept as-is rather than
adjusted to look tidier.)*

```python
docs = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "cats and dogs are great pets",
]
```

## 1. Bag-of-words

```python
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(docs)
print(vectorizer.get_feature_names_out())
print(X.toarray())
```

"the" has the highest raw count (appears twice in each of the first two
documents) — exactly the "common words dominate raw counts" problem
`01_concepts.md` describes.

## 2. From-scratch TF-IDF and the IDF=0 case

```python
import math
from collections import Counter

tokenized = [d.split() for d in docs]
N = len(docs)
vocab = sorted(set(w for d in tokenized for w in d))
df = {w: sum(1 for d in tokenized if w in d) for w in vocab}

def tfidf_full(doc_tokens):
    tf = Counter(doc_tokens)
    return [tf.get(w, 0) * math.log(N / (1 + df[w])) for w in vocab]

for i, d in enumerate(tokenized):
    print(i, dict(zip(vocab, tfidf_full(d))))
```

`"the"`, `"sat"`, `"on"` each appear in exactly 2 of the 3 documents, so
`IDF = log(3 / (1+2)) = log(1) = 0` **exactly** — a clean case where the
formula's denominator `(1 + document_count)` exactly equals `N`, zeroing
the log. Any word appearing in exactly `N-1` documents (out of `N`, with
this particular smoothed formula) will always get IDF exactly 0.

## 3. Compare to sklearn

```python
from sklearn.feature_extraction.text import TfidfVectorizer

sklearn_tfidf = TfidfVectorizer(smooth_idf=True, norm=None)
X_sklearn = sklearn_tfidf.fit_transform(docs)
print(X_sklearn.toarray())
```

Raw values differ slightly (sklearn's smoothed IDF formula
`log((1+N)/(1+df)) + 1` avoids ever hitting exactly 0, among other
differences), but the **ranking** of which words matter most per document
should agree with the from-scratch version — both should rate `"cat"` and
`"mat"` as document 0's most distinctive words.

## 4. Cosine similarity — a genuinely revealing result

```python
def cos_sim(a, b):
    dot = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(y*y for y in b))
    return dot / (na*nb) if na and nb else 0.0

vecs = [tfidf_full(d) for d in tokenized]
for i in range(3):
    for j in range(i+1, 3):
        print(i, j, cos_sim(vecs[i], vecs[j]))
```

**Actual output: all three pairwise similarities are exactly `0.0`.**

This is not a bug — it's TF-IDF's core limitation, laid bare: documents 0
("the cat sat on the mat") and 1 ("the dog sat on the log") are obviously
the *most* similar to a human (near-identical sentence structure), but
their shared words (`"the"`, `"sat"`, `"on"`) got IDF **zero** in Q2, and
their remaining words (`cat`/`mat` vs `dog`/`log`) share **no tokens at
all**. TF-IDF has no notion that "cat" and "dog" are related concepts (both
animals) — it only counts exact vocabulary overlap. This directly motivates
Lesson 057: **word embeddings** would place "cat" and "dog" near each
other in vector space (both animals, both often subjects of similar
sentences), correctly capturing a similarity that pure TF-IDF structurally
cannot see.

## 5. N-grams

```python
bigram_vectorizer = CountVectorizer(ngram_range=(1, 2))
X_bigram = bigram_vectorizer.fit_transform(docs)
print(len(bigram_vectorizer.get_feature_names_out()), "features (vs", len(vectorizer.get_feature_names_out()), "unigram-only)")
```

Vocabulary roughly doubles or more once bigrams are included (every
adjacent word pair becomes a candidate feature) — a feature like
`"sat on"` captures a specific local phrase pattern a unigram
representation splits into two independent, less-specific signals.

## 6. CountVectorizer vs TfidfVectorizer on real spam data

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv("../../../projects/003_spam-review-detector/02_solutions/data/sms_spam.csv")
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, stratify=df["label"], random_state=0)

for name, vec_cls in [("Count", CountVectorizer), ("TFIDF", TfidfVectorizer)]:
    vec = vec_cls()
    Xtr, Xte = vec.fit_transform(X_train), vec.transform(X_test)
    acc = MultinomialNB().fit(Xtr, y_train).score(Xte, y_test)
    print(name, acc)
```

On a dataset like Project 003's, where spam/ham vocabulary is already
fairly distinct (Lesson 030's discussion), both representations typically
perform similarly well — TF-IDF's down-weighting of common words matters
more on corpora where informative words are otherwise drowned out by very
frequent, low-signal terms; on cleanly-separable text, the choice between
`Count` and `TFIDF` features often matters less than the choice of model.
