# 01 — Requirement: RAG-Powered Assistant

## The brief

Build a question-answering assistant that only answers using information
retrieved from your own document collection — not the underlying LLM's
general pretrained knowledge alone.

## What to produce

1. **Ingestion pipeline**: load a folder of `.txt` documents, chunk them
   (Lesson 076), and build an embedding index — start with TF-IDF
   (Lesson 056, no extra dependencies), then upgrade to real dense
   embeddings (`sentence-transformers`) as a documented improvement.

2. **Retrieval**: given a query, retrieve the top-k most relevant chunks
   (cosine similarity, Lesson 010). Make `k` configurable.

3. **Generation**: construct a grounded prompt (Lesson 076's template) and
   send it to an LLM — a local model (Project 013's own GPT, or a small
   HF model) or an API-based one, your choice, but the prompt-construction
   and retrieval code should work regardless of which LLM ultimately
   consumes it.

4. **A "wrong questions" test set**: write at least 5 questions your
   document collection **cannot** answer (genuinely out of scope). Confirm
   your system says so explicitly (per Lesson 076's prompt instruction),
   rather than falling back on the LLM's general knowledge and answering
   anyway — this is often the single most important behavior to get right
   for a trustworthy RAG system.

5. **Retrieval evaluation** (Lesson 073): for at least 10 questions with
   known-correct source chunks, measure retrieval **recall@k** (is the
   correct chunk within the top-k retrieved, for k=1, 3, 5?). Report this
   separately from end-to-end answer quality — Lesson 076 explicitly
   warned these are two different things to measure.

6. **Compare TF-IDF vs dense embedding retrieval** on the same evaluation
   set from Q5. Does recall@k improve with dense embeddings, especially
   for questions phrased differently than the source documents (Lesson
   076's paraphrase-sensitivity finding)?

## Constraints

- The system must have a real "I don't know" / "not in my documents"
  behavior, tested and demonstrated, not just claimed.
- Don't peek at `02_solutions/` before you have a working pipeline and
  your own recall@k numbers.
