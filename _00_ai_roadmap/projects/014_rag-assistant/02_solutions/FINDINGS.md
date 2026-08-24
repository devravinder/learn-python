# Findings — RAG-Powered Assistant

*(Every number below is verified — actually computed by running
`evaluate_retrieval.py` against the real 8-document sample set in
`data/`. The generation step itself is a stub — see `rag_pipeline.py`'s
`answer_query` — since it needs a real LLM plugged in, which this
sandbox couldn't run; the retrieval half, which is the actual novel
engineering in a RAG system, is fully real and verified.)*

## Recall@k: easy vs. paraphrased queries

| Query difficulty | recall@1 | recall@3 | recall@5 |
|---|---|---|---|
| Easy (shares source vocabulary) | 100.0% | 100.0% | 100.0% |
| Hard (paraphrased, no shared vocabulary) | 75.0% | 75.0% | 75.0% |

All 8 easy queries retrieve their correct source document at rank 1. The 8
paraphrased queries drop to 75% (6/8) — and **recall doesn't improve at
all from k=1 to k=5**, meaning the 2 failing queries (about the Eiffel
Tower and honeybees, phrased without the source documents' vocabulary)
don't even show up in the top 5 out of 8 total documents. This directly
confirms Lesson 076's core limitation: TF-IDF retrieval is a genuine,
real weak point for realistically-phrased user questions, not just an
abstract concern — exactly the gap dense embeddings are meant to close.

## The out-of-scope threshold problem — a real, honest limitation

```text
in-scope top-1 similarities:     0.257 - 0.397
out-of-scope top-1 similarities: 0.054 - 0.338
```

**These ranges overlap substantially.** "What is the boiling point of
mercury?" (genuinely out of scope) scores **0.338** — higher than several
genuinely in-scope queries (Python: 0.257, Saturn: 0.264, Photosynthesis:
0.266). **A single fixed similarity threshold cannot cleanly separate
in-scope from out-of-scope queries for this corpus** — any threshold low
enough to accept the weakest true in-scope query (0.257) would also
incorrectly accept 3 of the 5 out-of-scope queries.

This is a genuinely important, verified finding, not a design flaw
specific to this implementation: naive similarity-threshold-based "I don't
know" detection is unreliable in general. More robust real-world
approaches include: (a) asking the LLM itself, given the retrieved
context, to judge whether the context actually answers the question
(shifting the judgment from a similarity score to the LLM's own
reasoning), or (b) training a dedicated relevance/entailment classifier on
labeled (query, context, relevant?) pairs — both meaningfully more
engineering than a threshold check, which is exactly why production RAG
systems invest in this rather than treating it as a one-line fix.

## Recommendation for this project's constraint (Q4 in the brief)

Given the threshold's unreliability, the more dependable "I don't know"
behavior is to **always include the instruction in the prompt** ("if the
context doesn't contain the answer, say so") and **rely on the LLM's own
judgment** at generation time, using the similarity threshold only as a
weak, secondary pre-filter (e.g. skip retrieval entirely only for
extremely low scores, well below anything seen in this evaluation) rather
than the primary mechanism for scope detection.

## What upgrading to dense embeddings would change

Based on Lesson 076's own verified finding (the same Eiffel Tower
paraphrase failed identically there), dense embeddings (`sentence-
transformers`) would be expected to recover the 2 failing hard queries
here too, since they place semantically related but lexically different
phrases (`"elevation"`/`"height"`, `"insects in a hive"`/`"honeybees"`)
near each other in vector space — recall@1 on the hard query set should
improve from 75% toward 100% with that upgrade, the single most impactful
change available to this system.
