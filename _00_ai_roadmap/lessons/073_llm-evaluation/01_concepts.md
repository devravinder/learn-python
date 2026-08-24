# 01 — Concepts: LLM Evaluation

## Held-out perplexity: the direct measure of language modeling quality

Lesson 063 introduced perplexity as `exp(cross-entropy loss)`. For genuine
evaluation, this must be computed on **held-out data the model never
trained on** — training-set perplexity only tells you how well the model
memorized what it saw (Lesson 017's overfitting concern, directly
applicable). Comparing perplexity **across different models** is only
valid when using the **same tokenizer and the same evaluation text** —
perplexity is sensitive to vocabulary/tokenization choices (Lesson 062:
different tokenizers segment the same text into different numbers of
tokens), so a "lower perplexity" claim comparing two differently-tokenized
models isn't a fair comparison.

```python
@torch.no_grad()
def compute_perplexity(model, eval_data, block_size):
    total_loss, total_tokens = 0.0, 0
    for i in range(0, len(eval_data) - block_size, block_size):
        x = eval_data[i:i+block_size].unsqueeze(0)
        y = eval_data[i+1:i+block_size+1].unsqueeze(0)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), reduction="sum")
        total_loss += loss.item()
        total_tokens += y.numel()
    return math.exp(total_loss / total_tokens)
```

## Beyond perplexity: task-specific benchmarks

Perplexity measures raw language-modeling quality but doesn't directly
measure *capability* at specific tasks (reasoning, factual knowledge,
code generation). Standard benchmark suites test this directly with
curated question sets and automatic scoring:

- **MMLU** (Massive Multitask Language Understanding): multiple-choice
  questions across 57 subjects (math, law, medicine, etc.) — a broad
  knowledge/reasoning benchmark.
- **HellaSwag**: commonsense reasoning — given a scenario, pick the most
  plausible continuation among several plausible-sounding distractors.
- **HumanEval**: code generation correctness, measured by actually
  **running** generated code against unit tests (Lesson 002-adjacent idea
  applied to evaluation itself).
- **TruthfulQA**: measures tendency to reproduce common misconceptions
  versus giving truthful answers.

```python
# using EleutherAI's lm-evaluation-harness (the de facto standard tool)
# lm_eval --model hf --model_args pretrained=your-model --tasks mmlu,hellaswag
```

## Multiple-choice evaluation, mechanically

For benchmarks like MMLU, the model isn't asked to *generate* an answer
letter directly — instead, evaluate the model's log-probability of each
candidate answer as a continuation of the question, and pick whichever
candidate the model assigns the highest probability to:

```python
def score_choice(model, tokenizer, question, choice):
    text = f"{question} {choice}"
    ids = tokenizer(text, return_tensors="pt")["input_ids"]
    logits = model(ids).logits
    log_probs = F.log_softmax(logits[:, :-1], dim=-1)
    choice_len = len(tokenizer(choice)["input_ids"])
    target_ids = ids[:, -choice_len:]
    token_log_probs = log_probs[:, -choice_len:].gather(-1, target_ids.unsqueeze(-1))
    return token_log_probs.sum().item()

best_choice = max(choices, key=lambda c: score_choice(model, tokenizer, question, c))
```

This connects directly back to Lesson 063's core objective — benchmark
scoring is fundamentally still "which continuation does the model consider
most probable," the same mechanism used for perplexity and generation.

## Why benchmark scores don't tell the whole story

- **Benchmark contamination**: if benchmark questions (or close
  paraphrases) leaked into the training data, scores are inflated and
  don't reflect genuine capability — a well-documented, ongoing concern
  for any model trained on large web-scraped corpora.
- **Benchmark saturation**: as models improve, older benchmarks can stop
  discriminating between "good" and "great" models (scores cluster near
  the ceiling) — motivating the continuous development of new, harder
  benchmarks.
- **Benchmarks measure what they measure**: high MMLU doesn't guarantee
  good behavior on your specific real use case — task-specific evaluation
  on data resembling your actual application is often more informative
  than any general benchmark leaderboard position.

## LLM-as-judge: using a strong model to evaluate another model's output

For open-ended tasks without a single correct answer (e.g. "is this
response helpful and well-written?"), a common modern approach is
prompting a strong LLM to rate or compare outputs — effectively
automating what Lesson 072's human raters did, at much lower cost and
higher scale, with the caveat that judge-model biases and blind spots
propagate into the evaluation. Used carefully (with human-rated
calibration checks), this has become a standard practical tool for
evaluating instruction-tuned/chat models at scale.

## Practical evaluation checklist for your own fine-tuned/trained models

1. Held-out perplexity, always, as a baseline sanity check.
2. A handful of relevant task-specific benchmarks if your model targets
   general capability.
3. **Manual qualitative review of real generated outputs** — no automatic
   metric fully replaces actually reading what your model produces,
   especially early in a project.
4. If comparing to a baseline (e.g. Project 012's classical vs. Transformer
   comparison), evaluate both on the **exact same** held-out data and
   metric, controlling everything else, precisely as Lesson 024 always
   emphasized for classical ML comparisons too.
