"""Generate a synthetic fallback text corpus, purely so the training
pipeline can be tested end-to-end before you source real text (a
public-domain book is strongly recommended for actually interesting
results - see the project README). Stdlib only.
"""
import random
from pathlib import Path

random.seed(0)

SUBJECTS = ["the old sailor", "a young queen", "the wandering scholar", "an ancient dragon",
            "the village blacksmith", "a curious child", "the forest spirit", "the last knight"]
VERBS = ["walked through", "discovered", "carefully studied", "quietly watched", "bravely entered",
         "slowly crossed", "finally reached", "secretly explored"]
OBJECTS = ["the misty forest", "an old stone tower", "the forgotten library", "a hidden valley",
           "the crumbling castle", "the silent river", "a sunlit meadow", "the deep cave"]
ENDINGS = ["and never looked back.", "wondering what came next.", "as the sun began to set.",
           "though the path ahead was unclear.", "with a growing sense of wonder.",
           "and found something unexpected.", "just as the rain began to fall.",
           "and remembered it for years to come."]


def generate_sentence():
    return f"{random.choice(SUBJECTS).capitalize()} {random.choice(VERBS)} {random.choice(OBJECTS)}, {random.choice(ENDINGS)}"


def generate_corpus(n_sentences=8000):
    paragraphs = []
    sentences = []
    for i in range(n_sentences):
        sentences.append(generate_sentence())
        if (i + 1) % random.randint(3, 6) == 0:
            paragraphs.append(" ".join(sentences))
            sentences = []
    if sentences:
        paragraphs.append(" ".join(sentences))
    return "\n\n".join(paragraphs)


if __name__ == "__main__":
    corpus = generate_corpus()
    out_path = Path(__file__).parent / "fallback_corpus.txt"
    out_path.write_text(corpus)
    print(f"Wrote {len(corpus)} characters ({len(corpus.encode('utf-8'))} bytes) to {out_path}")
