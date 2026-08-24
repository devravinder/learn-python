# 01 — Requirement: Build Your Own LLM End-to-End

## The brief

Assemble a complete pipeline: pretrain your own GPT (Project 013), teach
it to follow simple instructions (Lesson 071), and serve it behind a chat
API — all four stages working together, with honest reporting at each
transition.

## What to produce

### Stage 1–2: Tokenizer + Pretraining (reuse Project 013 directly)

1. If you don't already have a trained Project 013 checkpoint, produce one
   now: pick a real corpus, size your model using Lesson 068's ratio,
   train the BPE tokenizer, pretrain the GPT. Report final train/val
   perplexity.

### Stage 3: Instruction fine-tuning on your own pretrained model

2. Write **20-50 instruction/response pairs** in your model's domain (if
   you pretrained on, say, fairy tales, write instructions like
   "Continue this story:" or "Write a short tale about..."; match the
   instructions to what your small model could plausibly have learned —
   don't ask a tiny model trained on fiction to answer factual trivia).

3. Implement Lesson 071's masked-loss fine-tuning **on top of your
   pretrained checkpoint** (not a HF model this time — your own GPT class,
   your own BPE tokenizer). Fine-tune for a modest number of epochs.

4. Compare generation **before vs. after** instruction fine-tuning on 3
   held-out instructions: does the fine-tuned model attempt the
   instruction-following format, where the base pretrained model would
   have just continued the text unstructured (Lesson 063a/069's exact
   distinction)?

### Stage 4: Serving

5. Build a minimal chat API: a single endpoint that accepts a user message
   and returns the model's generated response (Lesson 066's `generate`,
   wrapped in a thin request/response layer — `http.server`, `Flask`, or
   `FastAPI`, your choice; Lesson 078 formalizes this properly, so a
   simple working version now is enough).

6. Write a tiny CLI or script client that sends a few messages to your
   running API and prints the responses — a full, working, if small,
   round trip from "user types a message" to "model responds," through an
   actual network request.

### Report

7. Write a short "model card" (a common real-world practice): your model's
   size, training data source and size, intended use case, and known
   limitations (be specific — not "it's small," but e.g. "struggles with
   factual questions outside its training domain," grounded in what you
   actually observed).

## Constraints

- Every component (tokenizer, model, training loop, fine-tuning, serving)
  should be code you wrote/assembled yourself across this curriculum — no
  wrapping a pretrained HF model in place of your own from Project 013.
- Don't peek at `02_solutions/` before you have your own working four-stage
  pipeline.
