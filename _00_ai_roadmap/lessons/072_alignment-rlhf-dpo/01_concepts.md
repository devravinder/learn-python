# 01 — Concepts: RLHF & DPO

## The gap instruction tuning (Lesson 071) leaves open

SFT trains on **fixed target responses** — the model learns to imitate
whatever responses were in the training data, with no signal about
*degrees* of quality among plausible responses. But "helpful," "safe," and
"well-reasoned" are graded, comparative qualities, not binary
correct/incorrect labels — exactly the kind of signal human **preferences**
(this response is *better* than that one) can provide, and fixed-target
imitation can't.

## Step 1: collect human preference data

Show human raters **two or more model responses** to the same prompt, ask
them to rank/pick the better one:

```
prompt: "Explain quantum entanglement to a 10-year-old."
response A: [a long, jargon-heavy explanation]
response B: [a short, clear analogy-based explanation]
human preference: B > A
```

This produces a dataset of `(prompt, chosen_response, rejected_response)`
triples — comparative, not absolute, labels.

## Step 2 (classic RLHF): train a reward model

Train a separate model to predict a scalar "quality score" that's
consistent with the collected human preferences — using the
**Bradley-Terry model** (a direct application of Lesson 006's probability
framework to pairwise comparisons):

```
P(chosen > rejected) = sigmoid(reward(chosen) - reward(rejected))
```

The reward model is trained (via ordinary cross-entropy, Lesson 016) to
make this probability high whenever humans actually preferred the
"chosen" response — after training, `reward(response)` gives a learned,
continuous quality estimate for *any* response, generalizing beyond the
exact examples humans rated.

## Step 3 (classic RLHF): optimize the policy against the reward model

Use policy gradient methods (Lesson 052 — specifically PPO, Proximal
Policy Optimization, a more stable variant not covered in depth here) to
fine-tune the language model (now playing the role of Lesson 050's "agent,"
with token-by-token generation as its "actions") to maximize the learned
reward model's score on its own generated responses:

```
objective = E[reward_model(response)] - β * KL(policy || reference_policy)
```

**The KL penalty term is essential** (Lesson 016's KL divergence, and
Lesson 054's own mention of it for the exact same reason): without it, the
policy would drift arbitrarily far from sensible language to exploit
quirks in the *learned* (imperfect) reward model — a well-documented
failure mode called **reward hacking**. The KL term keeps the fine-tuned
policy's output distribution anchored close to the original (pre-RLHF)
model's distribution, allowed to shift toward higher-reward behavior only
gradually and with a penalty for drifting too far.

## DPO: skipping the reward model and PPO entirely

**Direct Preference Optimization** (Rafailov et al., 2023) is a
mathematically clever simplification: it shows that the RLHF objective
above has a **closed-form** optimal policy in terms of the reward function,
which can be substituted back into the Bradley-Terry preference model to
get a loss **directly in terms of the policy itself** — no separate reward
model, no reinforcement learning/PPO required at all:

```
DPO loss = -log(sigmoid(β * [log(π(chosen)/π_ref(chosen)) - log(π(rejected)/π_ref(rejected))]))
```

This is trained with **ordinary supervised gradient descent** (Lesson
015) directly on the `(prompt, chosen, rejected)` preference triples —
much simpler to implement and tune than full RLHF's three-stage pipeline
(SFT → reward model → PPO), while achieving broadly comparable results in
practice. This is why DPO has become a very popular alternative to
classic RLHF for practical alignment fine-tuning.

## Why this connects back to Module 8

Notice the structure: **Module 8** taught you an agent (chess bot) learning
from a reward signal via policy gradients and self-play. **RLHF** is
literally the same framework — a "reward model" standing in for a
hand-crafted evaluation function, "generating token by token" standing in
for "choosing a move," policy gradients optimizing the policy against that
reward — applied to language instead of chess. If Lesson 052/054 made
sense, RLHF's mechanics (setting DPO's simplification aside) are not new
concepts, just a new application domain for ideas you've already built and
verified yourself.

## Practical guidance for Project-scale work

Full RLHF (reward model + PPO) is complex to implement correctly and
usually not worth attempting at hobby scale — DPO is the far more
practical choice if you want to experiment with preference-based alignment
yourself, since it only needs a preference dataset and standard supervised
training, no separate RL infrastructure. Libraries like Hugging Face's
`trl` provide ready-made `DPOTrainer` implementations for exactly this.

## What alignment fundamentally trades off

Both RLHF and DPO optimize toward what human raters *preferred* in the
training data — which means alignment quality is only as good as the
preference data's coverage and the raters' judgment. This is a real,
ongoing limitation (and active research area): a model can become very
good at satisfying the *specific* preferences it was trained on while
still failing in ways those particular comparisons never tested for —
worth knowing honestly rather than treating "aligned" as a solved,
binary property.
