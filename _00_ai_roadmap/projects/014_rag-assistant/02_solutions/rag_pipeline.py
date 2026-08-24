"""A complete, dependency-free RAG pipeline: chunking, TF-IDF retrieval,
grounded prompt construction, and an "out of scope" threshold. Swap in
`sentence-transformers` embeddings (see `dense_retrieval.py`) or a real
LLM call for `generate_answer` to complete the system end to end.
"""
import math
import re
from collections import Counter
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
OUT_OF_SCOPE_THRESHOLD = 0.05   # below this top similarity, treat as "not in my documents"


def load_documents(data_dir=DATA_DIR):
    docs = {}
    for path in sorted(data_dir.glob("*.txt")):
        docs[path.stem] = path.read_text().strip()
    return docs


def chunk_text(text, chunk_size=40, overlap=10):
    words = text.split()
    if len(words) <= chunk_size:
        return [text]
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
        if i + chunk_size >= len(words):
            break
    return chunks


def build_index(docs):
    """Returns (chunks, chunk_sources, vocab, df) - a simple TF-IDF index."""
    chunks, chunk_sources = [], []
    for doc_name, text in docs.items():
        for chunk in chunk_text(text):
            chunks.append(chunk)
            chunk_sources.append(doc_name)

    tokenized = [tokenize(c) for c in chunks]
    n = len(chunks)
    vocab = sorted(set(w for c in tokenized for w in c))
    df = {w: sum(1 for c in tokenized if w in c) for w in vocab}
    chunk_vecs = [tfidf_vec(c, vocab, df, n) for c in tokenized]
    return chunks, chunk_sources, vocab, df, chunk_vecs, n


def tokenize(text):
    return re.sub(r"[^\w\s]", "", text.lower()).split()


def tfidf_vec(tokens, vocab, df, n):
    tf = Counter(tokens)
    return [tf.get(w, 0) * math.log(n / (1 + df[w])) for w in vocab]


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def retrieve(query, chunks, chunk_sources, vocab, df, chunk_vecs, n, k=3):
    qvec = tfidf_vec(tokenize(query), vocab, df, n)
    sims = [cosine_similarity(qvec, cv) for cv in chunk_vecs]
    ranked = sorted(range(len(chunks)), key=lambda i: -sims[i])
    return [(chunks[i], chunk_sources[i], sims[i]) for i in ranked[:k]]


def build_prompt(query, retrieved_chunks):
    context = "\n\n".join(f"[{src}]: {chunk}" for chunk, src, _ in retrieved_chunks)
    return f"""Answer the question using ONLY the context below. If the context doesn't contain the answer, respond with "I don't have information about that in my documents."

Context:
{context}

Question: {query}
Answer:"""


def answer_query(query, index, k=3):
    chunks, chunk_sources, vocab, df, chunk_vecs, n = index
    retrieved = retrieve(query, chunks, chunk_sources, vocab, df, chunk_vecs, n, k=k)
    top_similarity = retrieved[0][2] if retrieved else 0.0

    if top_similarity < OUT_OF_SCOPE_THRESHOLD:
        return "I don't have information about that in my documents.", retrieved

    prompt = build_prompt(query, retrieved)
    # Plug in a real LLM call here, e.g.:
    #   return generate_with_llm(prompt), retrieved
    return f"[PROMPT BUILT - plug in an LLM here]\n{prompt}", retrieved


if __name__ == "__main__":
    docs = load_documents()
    index = build_index(docs)

    print("=== In-scope query ===")
    answer, retrieved = answer_query("How tall is the Eiffel Tower?", index)
    print("Top retrieved source:", retrieved[0][1], f"(sim={retrieved[0][2]:.3f})")

    print("\n=== Out-of-scope query ===")
    answer2, retrieved2 = answer_query("What is the capital of Australia?", index)
    print(answer2)
    print("(top similarity was:", round(retrieved2[0][2], 3) if retrieved2 else None, ")")
