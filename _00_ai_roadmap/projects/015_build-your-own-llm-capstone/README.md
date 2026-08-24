# Project 015 (Capstone) — Build Your Own LLM End-to-End

**Builds on:** Project 013, Modules 12–14 in full
**Difficulty:** Advanced (capstone)
**Estimated time:** 8–15 hours (plus training time)

## Objective

The final integration project for the entire "build your own LLM" arc:
**tokenizer → pretrain → instruction fine-tune → serve via a chat API** —
one coherent pipeline, built entirely from pieces you've already
implemented and verified yourself across Modules 9–14. Nothing here is
conceptually new; the work is assembling everything correctly into a
single, working system and being honest about what it can and can't do.

## The four stages, and where each one came from

| Stage | Reuses |
|---|---|
| 1. Tokenizer | Lesson 068a's from-scratch BPE tokenizer |
| 2. Pretrain | Project 013's GPT architecture + training loop |
| 3. Instruction fine-tune | Lesson 071's masked-loss SFT, applied to your own pretrained model (Lesson 070's LoRA optional, for practice) |
| 4. Serve | A minimal chat API (Lesson 075's serving concepts, previewing Lesson 078's FastAPI treatment) |

## Contents

1. [01_requirement.md](01_requirement.md)
2. [02_solutions/](02_solutions/)

## A note on scope, one more time

Lesson 068's scaling laws and Project 013's own findings already showed
that hobby-scale pretraining produces a model many orders of magnitude
smaller than any commercial LLM. This capstone does not change that — the
achievement here is a **complete, correct, working pipeline** you built
and understand end to end, not a model that competes with production
systems.
