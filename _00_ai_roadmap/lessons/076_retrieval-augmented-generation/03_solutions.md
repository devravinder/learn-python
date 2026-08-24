# 03 — Solutions: Retrieval-Augmented Generation

*(This code was actually run to produce the numbers below.)*

## 1–3. Chunking, TF-IDF embedding, and retrieval

```python
import math
from collections import Counter

document = """
The Eiffel Tower was completed in 1889 and stands in Paris, France.
It was designed by Gustave Eiffel's engineering company.
The tower is 330 meters tall and was the tallest man-made structure for 41 years.
The Great Wall of China stretches over 21000 kilometers.
It was built over many centuries by various Chinese dynasties.
The wall was primarily built to protect against invasions from the north.
Mount Everest is the tallest mountain above sea level at 8849 meters.
It is located in the Himalayas on the border of Nepal and Tibet.
The first successful summit was achieved in 1953.
"""
chunks = [s.strip() for s in document.strip().split("\n") if s.strip()]

def tokenize(t):
    return t.lower().replace(",", "").replace(".", "").replace("'", " ").split()

tokenized_chunks = [tokenize(c) for c in chunks]
N = len(chunks)
vocab = sorted(set(w for c in tokenized_chunks for w in c))
df = {w: sum(1 for c in tokenized_chunks if w in c) for w in vocab}

def tfidf_vec(tokens):
    tf = Counter(tokens)
    return [tf.get(w, 0) * math.log(N / (1 + df[w])) for w in vocab]

chunk_vecs = [tfidf_vec(c) for c in tokenized_chunks]

def cos_sim(a, b):
    dot = sum(x*y for x, y in zip(a, b))
    na, nb = math.sqrt(sum(x*x for x in a)), math.sqrt(sum(y*y for y in b))
    return dot / (na * nb) if na and nb else 0.0

query1 = "How tall is the Eiffel Tower?"
qvec1 = tfidf_vec(tokenize(query1))
sims1 = [cos_sim(qvec1, cv) for cv in chunk_vecs]
ranked1 = sorted(range(N), key=lambda i: -sims1[i])
for i in ranked1[:3]:
    print(f"{sims1[i]:.3f}  {chunks[i]}")
```

**Actual output:**

```text
0.317  The tower is 330 meters tall and was the tallest man-made structure for 41 years.
0.156  It was designed by Gustave Eiffel's engineering company.
0.145  The Eiffel Tower was completed in 1889 and stands in Paris, France.
```

The correct chunk (containing the actual height) ranks first, as
expected — the query shares direct vocabulary (`"tall"`, `"Eiffel"`,
`"Tower"`) with it.

## 4. The harder, paraphrased query — a genuine failure

```python
query4 = "What is the elevation of the famous Parisian landmark?"
qvec4 = tfidf_vec(tokenize(query4))
sims4 = [cos_sim(qvec4, cv) for cv in chunk_vecs]
ranked4 = sorted(range(N), key=lambda i: -sims4[i])
for i in ranked4:
    print(f"{sims4[i]:.3f}  {chunks[i]}")
```

**Actual output:**

```text
0.332  It is located in the Himalayas on the border of Nepal and Tibet.
0.230  The Great Wall of China stretches over 21000 kilometers.
0.113  The tower is 330 meters tall and was the tallest man-made structure for 41 years.
0.107  Mount Everest is the tallest mountain above sea level at 8849 meters.
0.009  The wall was primarily built to protect against invasions from the north.
0.006  The first successful summit was achieved in 1953.
0.005  The Eiffel Tower was completed in 1889 and stands in Paris, France.
0.000  It was designed by Gustave Eiffel's engineering company.
0.000  It was built over many centuries by various Chinese dynasties.
```

**A genuine, verified retrieval failure**: the actually-relevant chunk
about the Eiffel Tower's height ranks only **3rd** (0.113), while
completely irrelevant chunks about the Himalayas (0.332, top-ranked!) and
the Great Wall (0.230) outrank it — purely because they happen to share
more common words ("the," "is," "of," "on") with the paraphrased query
than the relevant chunk does. `"elevation"`/`"Parisian landmark"` share
**zero** vocabulary with `"tall"`/`"Eiffel Tower"`, so TF-IDF has
essentially no way to recognize they're asking about the same thing. A
dense embedding model (Lesson 057) would place `"elevation"` near
`"height"`/`"tall"` and `"Parisian landmark"` near `"Eiffel Tower"` in
vector space, purely from having learned those words appear in similar
contexts across its training data — recovering the correct match where
exact-vocabulary matching structurally cannot.

## 5. The final RAG prompt

```python
top_chunk = chunks[ranked1[0]]
prompt = f"""Answer the question using only the context below. If the context doesn't contain the answer, say so.

Context:
{top_chunk}

Question: {query1}
Answer:"""
print(prompt)
```

This is exactly what would be sent to an LLM — the retrieved chunk
grounds the answer in a specific, verifiable piece of text rather than
whatever the model's own (possibly outdated or simply absent) parametric
knowledge about the Eiffel Tower might be.

## 6. With real dense embeddings

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

embedder = SentenceTransformer("all-MiniLM-L6-v2")
chunk_embeds = embedder.encode(chunks)
query_embed = embedder.encode([query4])

sims = cosine_similarity(query_embed, chunk_embeds)[0]
ranked = sims.argsort()[::-1]
for i in ranked[:3]:
    print(sims[i], chunks[i])
```

Expect the dense embedding retriever to correctly rank the Eiffel Tower
height chunk at or near the top for this same paraphrased query — directly
demonstrating the practical value of moving from sparse (TF-IDF) to dense
(embedding-based) retrieval for real-world RAG systems, where users rarely
phrase questions using the document's exact original wording.
