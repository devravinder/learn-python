# 02 — Practicals: Retrieval-Augmented Generation

## A complete mini-RAG retrieval pipeline (pure Python — no dependencies)

Use this tiny "document" (a stand-in for a real knowledge base):

```python
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
```

1. Implement `chunk_text` (per `01_concepts.md`, small chunk size like 20
   words for this short example) and split the document into chunks.

2. Implement a simple TF-IDF-style vectorizer from scratch (reuse Lesson
   056's approach) over the chunks. Embed the query
   `"How tall is the Eiffel Tower?"` using the same vocabulary/IDF values.

3. Compute cosine similarity (Lesson 010) between the query vector and
   every chunk vector. Confirm the top-ranked chunk is genuinely the one
   containing the Eiffel Tower's height — a real, verifiable retrieval
   correctness check, not just "some chunk came back."

4. Try a **harder** query that avoids the relevant chunk's exact vocabulary:
   `"What is the elevation of the famous Parisian landmark?"` (asking
   about the Eiffel Tower's height without using "tall," "meters," "Eiffel,"
   or "tower"). Does TF-IDF-based retrieval still rank the correct chunk
   first — or does it get outranked by chunks that happen to share more
   common words, even though they're about the wrong landmark entirely?
   This is exactly Lesson 056/057's "TF-IDF has no notion of synonymy"
   limitation showing up concretely in a retrieval context — explain why a
   dense embedding-based retriever (Lesson 057) would likely do better.

5. Construct the final RAG prompt (per `01_concepts.md`'s template) using
   the top-1 retrieved chunk for the *first* query (Q3). Print the full
   prompt that would be sent to an LLM.

## With real dense embeddings (if you have `sentence-transformers`)

6. Repeat Q2-Q4 using `SentenceTransformer("all-MiniLM-L6-v2")` embeddings
   instead of TF-IDF. Does the harder query from Q4 now retrieve the
   correct chunk, where TF-IDF failed?
