# Reference Solutions

```bash
pip install transformers peft
python lora_finetune.py
```

*(Not independently executed here — no `transformers`/`peft`/PyTorch in
the authoring sandbox. Code follows `peft`'s documented API and Lesson
070/071's mechanics carefully; run it yourself to get real output.)*

## Expected behavior

- `model.print_trainable_parameters()` should report well under 1% of
  total parameters as trainable, consistent with Lesson 070's LoRA
  parameter-count findings.
- After fine-tuning on ~20 pirate-persona examples for several epochs,
  the 3 test prompts should show a clear shift toward pirate-speak
  phrasing, even though none of the 3 exact test prompts appeared in
  training — a real (if modest, given only ~20 training examples and a
  tiny base model) generalization within the training distribution's
  style.
- The out-of-domain generalization test (Q6, a science-explanation
  prompt very different from the training examples' topics) is a genuine
  open question: with only ~20 short training examples, the persona shift
  may be weaker or absent for topics far outside anything the model saw
  paired with that persona during training — report whatever you actually
  observe. A real "it didn't fully generalize" result here is a valid,
  expected finding at this data scale, not evidence something is broken.

For a more robust persona shift across topics, expand `TRAINING_EXAMPLES`
to 50-100+ diverse-topic pairs — the persona should generalize more
reliably as training data variety increases, directly illustrating the
same data-coverage point from Lesson 072's alignment discussion.
