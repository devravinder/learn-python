# 02 — Practicals: RLHF & DPO

## Bradley-Terry reward modeling (pure Python)

1. Implement the Bradley-Terry preference probability:
   `P(chosen > rejected) = sigmoid(reward(chosen) - reward(rejected))`.
   For toy scalar "rewards" `reward(A)=2.0, reward(B)=0.5`, compute
   `P(A preferred over B)`. Does it make sense that this comes out well
   above 0.5?

2. Implement the Bradley-Terry loss (`-log(P(chosen preferred))`) and
   compute the gradient of this loss with respect to `reward(chosen)` and
   `reward(rejected)` by hand (or via Lesson 013's numerical gradient
   check) for a case where the reward model currently has it backwards
   (`reward(chosen) < reward(rejected)`). Confirm the gradient pushes
   `reward(chosen)` up and `reward(rejected)` down — exactly what training
   the reward model on this preference pair should do.

## DPO loss (pure Python)

3. Implement the DPO loss from `01_concepts.md` using plain scalars for
   `log(π(chosen)/π_ref(chosen))` and `log(π(rejected)/π_ref(rejected))`
   (pretend these log-ratios are given, e.g. `2.0` and `-1.0`
   respectively — meaning the current policy already favors "chosen" more
   than the reference policy does, and disfavors "rejected" more). Compute
   the loss at `β=0.1` and `β=1.0`. Does higher `β` make the loss more or
   less sensitive to a given gap between the two log-ratios?

4. Compute the DPO loss for the **opposite** scenario (log-ratios `-2.0`
   and `1.0` — the policy has drifted the *wrong* way, favoring "rejected"
   over what the reference model would). Confirm the loss is much higher
   than in Q3, and that gradient descent on this loss would need to push
   the policy back in the correct direction.

## Reflection

5. Explain in your own words why the **KL penalty term** in the RLHF
   objective (`01_concepts.md`) is necessary — what specifically could go
   wrong during PPO fine-tuning if you optimized purely for the reward
   model's score with no such penalty at all? (This is "reward hacking" —
   look up one real documented example if you're not immediately sure what
   form it takes.)

6. DPO's loss (Q3-Q4) is computed directly from log-probability ratios
   under the current and reference policy — no reward model, no RL
   rollout, no PPO. Given what you now understand about both approaches,
   write 3-4 sentences on why DPO is considered dramatically simpler to
   implement correctly than full RLHF, referencing specifically what
   machinery DPO doesn't need that RLHF does.
