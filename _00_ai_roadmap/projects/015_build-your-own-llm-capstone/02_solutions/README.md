# Reference Solution

```bash
pip install fastapi uvicorn requests

# Stage 1-2 (reuse Project 013 directly):
cd ../../013_train-your-own-gpt/02_solutions
python data/generate_fallback_corpus.py    # or use your own real corpus
python train.py --data data/fallback_corpus.txt --max_steps 3000

# Stage 3:
cd ../../015_build-your-own-llm-capstone/02_solutions
python finetune_instructions.py --checkpoint ../../013_train-your-own-gpt/02_solutions/gpt_checkpoint.pt \
    --data ../../013_train-your-own-gpt/02_solutions/data/fallback_corpus.txt

# Stage 4 (in one terminal):
python serve_api.py --data ../../013_train-your-own-gpt/02_solutions/data/fallback_corpus.txt
# (in another terminal):
python client.py "Continue this story:"
```

- [finetune_instructions.py](finetune_instructions.py) — Stage 3: masked-
  loss instruction fine-tuning on your own pretrained GPT (Lesson 071,
  applied to Project 013's model instead of a HF model)
- [serve_api.py](serve_api.py) — Stage 4: a minimal FastAPI chat endpoint
  wrapping Lesson 066's generation loop
- [client.py](client.py) — proves the round trip actually works over a
  real HTTP request
- [FINDINGS.md](FINDINGS.md) — a model-card template and expected
  before/after fine-tuning behavior (not independently executed here — no
  PyTorch/FastAPI in the authoring sandbox; every reused piece — tokenizer,
  masking arithmetic, training loop — was independently verified in its
  original lesson/project)

Try [01_requirement.md](../01_requirement.md) yourself first. Write your
own model card (Q7) honestly — the specific, concrete limitations you
actually observe are more valuable than a generic "it's a small model"
disclaimer.
