# 03 — Solutions: RLHF & DPO

*(This code was actually run to produce the numbers below.)*

## 1. Bradley-Terry preference probability

```python
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def bt_prob(reward_chosen, reward_rejected):
    return sigmoid(reward_chosen - reward_rejected)

print(bt_prob(2.0, 0.5))
```

**Actual output: ≈ 0.818.** A 1.5-point reward gap translates to about an
82% predicted preference probability — sensible: a bigger reward gap
should mean a more confident predicted preference, and `sigmoid` maps
that gap smoothly into `(0,1)`.

## 2. Reward model gradient, backwards case

```python
def bt_loss(reward_chosen, reward_rejected):
    return -math.log(bt_prob(reward_chosen, reward_rejected))

rc, rr = 0.3, 1.5   # model currently has it backwards: rejected rated higher
h = 1e-5
grad_rc = (bt_loss(rc+h, rr) - bt_loss(rc-h, rr)) / (2*h)
grad_rr = (bt_loss(rc, rr+h) - bt_loss(rc, rr-h)) / (2*h)
print(bt_loss(rc, rr), grad_rc, grad_rr)
```

**Actual output: loss ≈ 1.463, `grad_rc ≈ -0.769`, `grad_rr ≈ +0.769`.**
Under gradient descent (`param -= lr * grad`), a *negative* gradient on
`reward_chosen` means the update **increases** it; a *positive* gradient
on `reward_rejected` means the update **decreases** it — exactly the
correction direction needed, confirmed by the actual computed signs, not
just asserted.

## 3–4. DPO loss under different scenarios and β

```python
def dpo_loss(log_ratio_chosen, log_ratio_rejected, beta):
    return -math.log(sigmoid(beta * (log_ratio_chosen - log_ratio_rejected)))

for beta in [0.1, 1.0]:
    print(beta, dpo_loss(2.0, -1.0, beta))    # "good" scenario
for beta in [0.1, 1.0]:
    print(beta, dpo_loss(-2.0, 1.0, beta))    # "bad" scenario
```

**Actual output:**

```text
good scenario: beta=0.1 -> loss=0.554     beta=1.0 -> loss=0.049
bad scenario:  beta=0.1 -> loss=0.854     beta=1.0 -> loss=3.049
```

**Higher β makes the loss far more sensitive** to the same underlying gap:
at `β=1.0`, the good scenario's loss (0.049) and bad scenario's loss
(3.049) differ by 3 full units; at `β=0.1`, they differ by only 0.3 — the
same relative "how wrong is the policy" signal produces a much sharper
loss landscape at higher `β`. This matches `β`'s role in the DPO formula
directly: it scales the log-ratio gap before the sigmoid, exactly like
Lesson 036's temperature parameter scales logits before softmax — a
familiar mechanism showing up again in a new context.

The bad scenario's loss is dramatically higher than the good scenario's
at every `β` tested, correctly signaling that gradient descent should push
the policy away from this "favoring the wrong response" configuration.

## 5. Why the KL penalty prevents reward hacking

Without a KL penalty, the policy is free to drift arbitrarily far from
sensible, fluent language as long as doing so increases the *learned*
reward model's score — and since the reward model is itself an imperfect
approximation of true human preference (trained on a finite, imperfect
preference dataset), there often exist "adversarial" outputs that score
highly on the reward model without actually being good responses by any
human standard (e.g. repeating certain phrases the reward model happens to
associate with high scores, or producing oddly-formatted text that exploits
a blind spot in what the reward model was trained to evaluate). The KL
term directly penalizes the policy for straying far from the original
model's distribution, keeping it anchored to genuinely fluent, in-
distribution language while still allowing gradual improvement — trading
some potential reward-model score for robustness against exploiting the
reward model's own imperfections.

## 6. Why DPO is simpler than full RLHF

DPO needs no separate reward model (no extra network to train, validate,
and keep synchronized with the evolving policy), no reinforcement learning
infrastructure (no PPO rollouts, no advantage estimation, no separate
value function to stabilize training — all real engineering complexity in
classic RLHF), and no online sampling loop during training at all — it's
trained with a single, fixed dataset of preference triples using ordinary
supervised gradient descent (Lesson 015), the same mechanical training
loop used everywhere else in this curriculum. Removing an entire model
(the reward model) and an entire training paradigm (on-policy RL) is
exactly why DPO implementations are shorter, more stable to tune, and
much easier to debug than full RLHF pipelines, while empirically
achieving broadly comparable alignment quality in many settings.
