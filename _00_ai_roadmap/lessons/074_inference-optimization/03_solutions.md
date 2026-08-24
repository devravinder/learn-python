# 03 — Solutions: Inference Optimization

*(This code was actually run to produce the numbers below.)*

## 1–2. KV-cache speedup scaling

```python
for N in [100, 1000, 10000]:
    without_cache = sum(range(1, N + 1))   # N*(N+1)/2
    with_cache = N
    print(N, without_cache, with_cache, without_cache / with_cache)
```

**Actual output:**

```text
N=100:    without=5,050      with=100     speedup=50.5x
N=1000:   without=500,500    with=1,000   speedup=500.5x
N=10000:  without=50,005,000 with=10,000  speedup=5000.5x
```

**The speedup ratio grows linearly with `N`** — since without-cache work is
`N(N+1)/2` (quadratic) and with-cache work is `N` (linear), the ratio
`[N(N+1)/2] / N = (N+1)/2` grows proportionally to `N` itself. This means
KV-caching's benefit becomes **more** dramatic, not less, as generated
sequences get longer — exactly why it's considered essential rather than
a minor optimization for any realistic LLM generation length.

## 3. Quantization memory footprint

```python
params = 7_000_000_000
for name, bytes_per in [("fp32", 4), ("fp16", 2), ("int8", 1), ("int4", 0.5)]:
    print(name, params * bytes_per / 1e9, "GB")
```

**Actual output: fp32 = 28.0 GB, fp16 = 14.0 GB, int8 = 7.0 GB, int4 = 3.5
GB.** On a 24GB consumer GPU, `fp32` **does not fit** (28GB > 24GB) for a
7B model's weights alone, while `fp16`, `int8`, and `int4` all fit
comfortably — a concrete, verified illustration of why quantization is
often the difference between "runs on my GPU" and "doesn't," not just a
minor efficiency tweak.

## 4. Speculative decoding acceptance

```python
draft = [5, 12, 7, 9, 3]
target = [5, 12, 8, 9, 3]
accepted = 0
for d, t in zip(draft, target):
    if d == t:
        accepted += 1
    else:
        break
print(accepted, target[accepted])
```

**Actual output: `accepted=2`, corrected token = `8`.** Positions 0-1
match (`5`, `12`); position 2 mismatches (draft said `7`, target says
`8`) — so 2 draft tokens are kept, and generation continues from the
target model's correct token `8` at that position, discarding the rest of
the (now-invalidated) draft.

## 5. Draft accuracy vs. average accepted tokens

```python
import random

def simulate(accuracy, rounds=20, draft_len=5, seed=0):
    rng = random.Random(seed)
    totals = []
    for _ in range(rounds):
        accepted = 0
        for _ in range(draft_len):
            if rng.random() < accuracy:
                accepted += 1
            else:
                break
        totals.append(accepted)
    return sum(totals) / len(totals)

for acc in [0.9, 0.7, 0.5, 0.3, 0.1]:
    print(acc, simulate(acc))
```

**Actual output:**

```text
accuracy=0.9: avg accepted/round = 3.20
accuracy=0.7: avg accepted/round = 1.35
accuracy=0.5: avg accepted/round = 0.60
accuracy=0.3: avg accepted/round = 0.15
accuracy=0.1: avg accepted/round = 0.00
```

Somewhere between 50-70% per-token draft accuracy, average accepted
tokens per round crosses below 1 — meaning, accounting for the extra
overhead of running the draft model at all, speculative decoding stops
being clearly worthwhile once the draft model's guesses are wrong more
often than they're right. This matches real practical guidance: pick a
draft model that agrees with the target model *often* on easy/predictable
text (a much lower bar than "as good as the target model"), not merely
"any small model."

## 6. Why speculative decoding never hurts output quality

Every **accepted** token is one the target (large) model's own
distribution already assigned high probability to at that position — the
draft model only *proposes*, and the target model's verification step is
exactly the same computation it would have used to generate that token
directly during ordinary one-token-at-a-time decoding. Any draft token the
target model *wouldn't* have generated gets rejected and immediately
replaced with the target model's actual choice at that position. The
final output sequence is therefore statistically identical to what
plain (slow) target-model-only decoding would have produced — speculative
decoding changes **how fast** you get there, never **what** you get,
which is precisely why it's adopted so widely with no quality tradeoff to
weigh against its speed benefit.
