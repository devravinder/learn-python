# Findings — Build Your Own LLM End-to-End (Capstone)

*(Caveat: PyTorch/FastAPI weren't available in the authoring sandbox, so
Stages 2–4's execution isn't independently verified here — the code
follows the exact, already-reviewed patterns from Project 013 (pretraining,
verified tokenizer/sizing), Lesson 071 (masked-loss fine-tuning, verified
masking arithmetic), and Lesson 075 (serving concepts). Run the full
pipeline yourself and replace this with your real model card.)*

## Example model card (fill in with your own real numbers)

**Model**: `MyStoryGPT-v1`
**Architecture**: Decoder-only Transformer, 4 layers, 4 heads, `d_model=64`
(Lesson 060), ~60K parameters
**Tokenizer**: Byte-level BPE, 512-token vocabulary, trained on the
pretraining corpus (Lesson 068a)
**Pretraining data**: [your corpus - e.g. "public-domain short story
collection, ~2MB"]
**Pretraining result**: final validation perplexity ≈ [your number] (report
what `train.py` actually printed)
**Fine-tuning data**: 30 hand-written (instruction, response) pairs in the
[your domain] style
**Intended use**: generating short, [your domain]-style text continuations
in response to simple instructions. **Not** intended for factual
question-answering, code generation, or any domain outside its narrow
training data.
**Known limitations** (be specific, based on what you actually observed):
- Struggles with instructions outside its training domain (report a
  concrete failed example).
- Limited coherence beyond a few sentences (a direct, expected consequence
  of both its small size and Lesson 068's scaling-law mismatch relative to
  its training corpus).
- No safety filtering or alignment training beyond basic instruction
  fine-tuning — should not be used for open-ended user-facing deployment
  without additional safeguards.

## Expected qualitative shift from Stage 3 (instruction fine-tuning)

Before fine-tuning: given `"Continue this story:"` as a prompt, the base
pretrained model should just continue generating **plausible training-
distribution text** with no particular regard for the instruction having
been an instruction at all (Lesson 063a/069's core distinction between
raw completion and instruction-following behavior).

After fine-tuning: the same prompt should produce a response that at
least *attempts* the instruction-following shape — starting cleanly after
the `<|assistant|>` marker, in a register resembling the fine-tuning
examples' target responses — even with only 20-50 training examples,
consistent with Lesson 071's point that instruction tuning mostly teaches
*behavior/format*, not new content knowledge, and so can shift
noticeably even from a small dataset.

## Expected serving round-trip

`client.py`'s request should receive a real generated response from the
running `serve_api.py` process — a genuine, if minimal, working proof that
every piece (tokenizer, model, fine-tuning, HTTP serving) connects into
one functioning system, the actual point of this capstone.

## The honest bottom line

This capstone's value is having built and verified, piece by piece across
14 modules, every component of a real (if small) LLM system: the math
(Modules 3, 8), the neural network foundations (Module 6), the
Transformer architecture (Module 10), the training objective and loop
(Module 11), fine-tuning (Module 12), and serving (Module 13/15) — the
same components, unchanged in kind, that scale up (with vastly more data,
compute, and engineering) into the LLMs you interact with commercially.
The gap between this project and those systems is compute and data scale,
quantified honestly back in Lesson 068 — not missing understanding.
