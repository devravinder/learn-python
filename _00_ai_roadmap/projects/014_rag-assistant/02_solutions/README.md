# Reference Solution

```bash
python rag_pipeline.py        # demo: in-scope + out-of-scope query
python evaluate_retrieval.py   # recall@k + threshold analysis (fully verified, see FINDINGS.md)
```

- [rag_pipeline.py](rag_pipeline.py) — chunking, TF-IDF retrieval, grounded
  prompt construction, out-of-scope threshold (dependency-free)
- [evaluate_retrieval.py](evaluate_retrieval.py) — recall@k on easy vs.
  paraphrased queries, and an in-scope-vs-out-of-scope similarity analysis
- [FINDINGS.md](FINDINGS.md) — **fully verified** results, including a
  genuinely important finding: a fixed similarity threshold cannot
  reliably separate in-scope from out-of-scope queries for this corpus

**To complete the system**: plug a real LLM call into `answer_query`'s
`generate_answer` placeholder (a local HF model, Project 013's own GPT, or
an API-based model) — the retrieval half is complete and evaluated;
generation is intentionally left as a one-line swap so you can use
whatever LLM you have access to.

**To upgrade retrieval**: replace `tfidf_vec`/`cosine_similarity` with
`sentence-transformers` embeddings — `FINDINGS.md` predicts exactly which
2 test queries should improve and why, so you have a concrete before/after
to verify once you make the swap.

Try [01_requirement.md](../01_requirement.md) yourself first.
