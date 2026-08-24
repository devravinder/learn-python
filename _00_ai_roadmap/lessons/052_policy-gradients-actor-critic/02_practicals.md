# 02 — Practicals: Policy Gradients

A 4-armed bandit (single state, no sequential decisions — the simplest
possible setting to see REINFORCE's core mechanics clearly): arms have
true win probabilities `[0.2, 0.5, 0.8, 0.3]` (arm 2 is best), each pull
returns reward `1` (win) or `0` (loss).

1. Implement a softmax policy over 4 logits (`01_concepts.md`'s
   `softmax`), and `sample_action(probs)` (sample an action according to
   the probabilities).

2. Implement the REINFORCE update for this stateless bandit: after
   sampling action `a` and observing reward `r`, the gradient of
   `log π(a)` with respect to each logit `i` is `onehot(a)[i] - probs[i]`
   (a standard, useful identity for softmax + log — worth taking as given
   here). Update: `logits[i] += alpha * r * grad[i]`.

3. Train for 2000 steps at `alpha=0.1`. Print the final softmax
   probabilities. Does the policy converge to strongly prefer arm 2 (the
   true best arm)?

4. Add a moving-average baseline (`baseline += 0.05 * (reward - baseline)`
   after each step) and use `(reward - baseline)` in place of raw `reward`
   in the update. Confirm this version also converges to preferring arm 2.

5. Compare the **variance** of convergence speed between the no-baseline
   and with-baseline versions: run each 300 times with different random
   seeds, recording how many steps it takes for `P(arm 2) >= 0.9`. Report
   the mean and standard deviation of steps-to-converge for both. Does the
   baseline reduce the mean, the variance, both, or neither here?

6. Explain in your own words why a reward of exactly `0` (a loss) produces
   **no gradient update at all** in the no-baseline version (look at the
   update rule: `alpha * r * grad`, and consider `r=0`), but *does* produce
   a (negative) update in the with-baseline version. Which behavior seems
   more sensible for learning from failures, and why?
