"""Evaluate retrieval quality (recall@k) separately from end-to-end answer
quality, per Lesson 073/076's guidance. Also probes the out-of-scope
similarity threshold's reliability.

Run:
    python evaluate_retrieval.py
"""
from rag_pipeline import build_index, load_documents, retrieve

# easy queries: share vocabulary with their source document
EASY_QUERIES = [
    ("How tall is the Eiffel Tower?", "doc1_eiffel_tower"),
    ("When was the Great Wall built?", "doc2_great_wall"),
    ("Who first climbed Mount Everest?", "doc3_everest"),
    ("Who created Python?", "doc4_python"),
    ("What gas do plants release during photosynthesis?", "doc5_photosynthesis"),
    ("How many bitcoins will ever exist?", "doc6_bitcoin"),
    ("How do bees communicate food locations?", "doc7_honeybees"),
    ("How many moons does Saturn have?", "doc8_saturn"),
]

# hard queries: deliberately paraphrased, avoiding the source document's exact vocabulary
HARD_QUERIES = [
    ("What is the elevation of the famous Parisian landmark?", "doc1_eiffel_tower"),
    ("Which fortification defended against northern raiders for centuries?", "doc2_great_wall"),
    ("Who reached the peak of the world's highest mountain first?", "doc3_everest"),
    ("Which scripting language did Guido van Rossum create?", "doc4_python"),
    ("What byproduct do plants release while making food from sunlight?", "doc5_photosynthesis"),
    ("What is the maximum supply of the cryptocurrency invented by Nakamoto?", "doc6_bitcoin"),
    ("How do insects in a hive signal where flowers are?", "doc7_honeybees"),
    ("Which ringed planet has a moon larger than Mercury?", "doc8_saturn"),
]

OUT_OF_SCOPE_QUERIES = [
    "What is the capital of Australia?",
    "What is the boiling point of mercury?",
    "Who won the 2020 US presidential election?",
    "What is the best recipe for chocolate cake?",
    "How do I fix a flat tire?",
]


def recall_at_k(queries, index, k):
    chunks, chunk_sources, vocab, df, chunk_vecs, n = index
    hits = 0
    for query, expected_doc in queries:
        results = retrieve(query, chunks, chunk_sources, vocab, df, chunk_vecs, n, k=k)
        if expected_doc in [r[1] for r in results]:
            hits += 1
    return hits / len(queries)


def main():
    docs = load_documents()
    index = build_index(docs)

    print("=== Recall@k: easy (exact-vocabulary) queries ===")
    for k in [1, 3, 5]:
        print(f"  recall@{k}: {recall_at_k(EASY_QUERIES, index, k):.1%}")

    print("\n=== Recall@k: hard (paraphrased) queries ===")
    for k in [1, 3, 5]:
        print(f"  recall@{k}: {recall_at_k(HARD_QUERIES, index, k):.1%}")

    print("\n=== Similarity scores: in-scope vs out-of-scope ===")
    chunks, chunk_sources, vocab, df, chunk_vecs, n = index
    print("in-scope (easy) top-1 similarities:")
    for q, _ in EASY_QUERIES:
        sim = retrieve(q, chunks, chunk_sources, vocab, df, chunk_vecs, n, k=1)[0][2]
        print(f"  {sim:.3f}  {q}")
    print("out-of-scope top-1 similarities:")
    for q in OUT_OF_SCOPE_QUERIES:
        sim = retrieve(q, chunks, chunk_sources, vocab, df, chunk_vecs, n, k=1)[0][2]
        print(f"  {sim:.3f}  {q}")


if __name__ == "__main__":
    main()
