# 03 — Solutions: Text Preprocessing & Tokenization

*(Q5's BPE trainer was actually run to produce the merges below.)*

## 1. Basic preprocessing

```python
import re

def basic_preprocess(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.split()

text = "Tokenization is the FIRST step in any NLP pipeline! It matters a lot."
tokens = basic_preprocess(text)
print(tokens, len(set(tokens)))
```

## 2. Crude rule-based stemming

```python
def crude_stem(word):
    for suffix in ["ing", "ed", "s"]:
        if word.endswith(suffix):
            return word[: -len(suffix)]
    return word

for w in ["running", "runs", "ran", "easily", "flies"]:
    print(w, "->", crude_stem(w))
```

`running -> runn` (not a real word — should be "run," exposing the crude
rule's imprecision), `runs -> run` (correct), `ran -> ran` (unchanged —
irregular verbs aren't handled by simple suffix-stripping at all),
`easily -> easily` (no matching suffix), `flies -> flie` (not a real word).
This is expected: real stemmers (e.g. the Porter stemmer) use many more
rules and still aren't perfect — lemmatization (using a real dictionary) is
more accurate but slower and needs a language model/vocabulary resource.

## 3. Character vs word tokenization

```python
paragraph = "Tokenization turns raw text into numbers a model can use. " * 3
word_tokens = paragraph.split()
char_tokens = list(paragraph)
print("words:", len(word_tokens), "chars:", len(char_tokens))
print("chars per word:", len(char_tokens) / len(word_tokens))
```

Typical English text runs roughly 5-6 characters per word (including the
trailing space) — so character-level tokenization produces sequences
roughly 5-6x longer than word-level for the same text, directly
illustrating the "much longer sequences" tradeoff from `01_concepts.md`.

## 4. Out-of-vocabulary simulation

```python
train_text = "the cat sat on the mat and looked at the small dog"
test_text = "the astronaut piloted a spacecraft toward the distant moon"

vocab = set(basic_preprocess(train_text))
test_tokens = basic_preprocess(test_text)
oov_tokens = [t for t in test_tokens if t not in vocab]

print(f"{len(oov_tokens)}/{len(test_tokens)} tokens are OOV: {oov_tokens}")
```

Since the test paragraph covers an entirely different topic (space vs
household pets), expect a **high OOV rate** (most content words unseen) —
only common function words like "the" survive, a direct, concrete
demonstration of word-level tokenization's fragility to any topic/vocabulary
shift, exactly the problem subword tokenization (Q5, and Lesson 068a) is
built to avoid.

## 5. Minimal BPE trainer

```python
from collections import Counter

text = "low lower lowest newer newest wider widest low low lower newer"
corpus = [list(w) + ["</w>"] for w in text.split()]

def get_pair_counts(corpus):
    counts = Counter()
    for word in corpus:
        for i in range(len(word) - 1):
            counts[(word[i], word[i+1])] += 1
    return counts

def merge_pair(corpus, pair):
    merged = "".join(pair)
    new_corpus = []
    for word in corpus:
        new_word, i = [], 0
        while i < len(word):
            if i < len(word) - 1 and (word[i], word[i+1]) == pair:
                new_word.append(merged)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_corpus.append(new_word)
    return new_corpus

vocab = set(c for w in corpus for c in w)
for step in range(10):
    counts = get_pair_counts(corpus)
    if not counts:
        break
    best_pair = max(counts, key=counts.get)
    corpus = merge_pair(corpus, best_pair)
    vocab.add("".join(best_pair))
    print(f"merge {step+1}: {best_pair} (count={counts[best_pair]})")
```

**Actual output:**

```text
merge 1: ('l', 'o')     count=6
merge 2: ('lo', 'w')    count=6
merge 3: ('e', 'r')     count=5
merge 4: ('er', '</w>') count=5
merge 5: ('low', '</w>') count=3
merge 6: ('e', 's')     count=3
merge 7: ('es', 't')    count=3
merge 8: ('est', '</w>') count=3
merge 9: ('n', 'e')     count=3
merge 10: ('ne', 'w')   count=3
```

The algorithm correctly discovers `"low"` and `"er"` as frequent chunks
first (they appear in 6 and 5 words respectively, more than any other
pair), then builds up `"lower"`, `"lowest"`, `"newer"` from smaller merged
pieces — exactly the classic behavior described in the original BPE paper
(Sennrich et al. 2016), whose own worked example uses this same
low/lower/lowest/newer/newest/wider/widest text. After 10 merges, common
words like "low" tokenize as a single unit (`['low</w>']`) while less
frequent combinations still split into pieces — precisely the "common
words stay whole, rare ones split" property that makes subword
tokenization practical.

## 6. Comparing pretrained tokenizers

```python
from transformers import AutoTokenizer

gpt2_tok = AutoTokenizer.from_pretrained("gpt2")
bert_tok = AutoTokenizer.from_pretrained("bert-base-uncased")

sentence = "Tokenization splits uncommonly-used words into subword pieces."
print("GPT-2:", gpt2_tok.convert_ids_to_tokens(gpt2_tok.encode(sentence)))
print("BERT: ", bert_tok.convert_ids_to_tokens(bert_tok.encode(sentence)))
```

GPT-2 uses byte-level BPE (operates on raw bytes, so it can represent any
Unicode text with no true OOV token at all, and often prefixes
continuation pieces with a special marker like `Ġ` for the preceding
space); BERT uses WordPiece (conceptually similar to BPE but merges based
on maximizing training-data likelihood rather than raw frequency, and
marks word-continuation pieces with a `##` prefix, e.g. `sub`, `##word`).
Both solve the same underlying OOV problem via different specific merge
criteria and conventions.
