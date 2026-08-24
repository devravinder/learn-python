# 03 — Solutions: ML System Design Interview Prep

These are model answers, not "the" answer — an interviewer wants to see
the reasoning and structure below, not this exact architecture. Q1 and Q4
are worked in full; Q2/Q3/Q5 give the key points an answer should hit;
Q6 is left to you since it depends on which prompt you picked.

## 1. Spam detection for a messaging app

1. **Clarify**: Real-time (score at send-time, before delivery), scale
   assumed at millions of messages/day, false positives (blocking a real
   message) are costly to user trust, so precision matters as much as
   recall.
2. **Frame**: Binary classification — `P(spam | message, sender history)`.
3. **Data**: User reports ("mark as spam") give noisy positive labels;
   need negative sampling from non-reported messages. Label delay:
   reports can arrive hours after send, so very recent messages lack
   ground truth — training data is inherently a few hours to days stale.
4. **Features**: Message text (Lesson 055/056 — TF-IDF or embeddings),
   sender account age, sending rate (messages/minute — a classic spam
   signal), recipient-report history for this sender. Watch for leakage:
   "number of reports this sender has received" must only count reports
   that existed *before* this message was sent, not the final total.
5. **Model**: Start with Naive Bayes/logistic regression on TF-IDF
   (Project 003's baseline) — fast, interpretable, cheap to serve at
   real-time latency. Escalate to a small Transformer classifier only if
   the baseline's recall on adversarial/obfuscated spam is insufficient.
6. **Offline eval**: Precision-recall curve, not accuracy (spam is a
   small minority class — Lesson 025) — pick the threshold matching the
   cost tradeoff from step 1.
7. **Serving**: Sub-100ms budget (blocking send flow) — favor the
   lightweight classifier; run at message-send time as a synchronous
   API call (Lesson 078's pattern).
8. **Online eval**: A/B test against current rules-based filter, measure
   report rate reduction and false-positive complaints post-launch.
9. **Monitoring**: Spammers adapt quickly (adversarial drift, faster than
   typical concept drift) — monitor recall proxy (report rate) daily, not
   just a scheduled monthly retrain; consider a fast-retrain pipeline.
10. **Tradeoff**: A heavier model might catch more obfuscated spam but
    risks the latency budget in step 7 — flag this explicitly rather than
    silently picking one side.

## 4. Chess Bot matchmaking/rating system

1. **Clarify**: Goal is fair, engaging matches — not a supervised
   prediction problem in the traditional sense.
2. **Frame**: This is a **ranking/rating estimation problem**, not
   classification — the right tool is an Elo/Glicko-style rating system
   (each player, human or bot, has a rating updated after each game based
   on the result vs. expectation), not a trained classifier. Recognizing
   "this isn't a classification problem, here's what it actually is" is
   itself the answer the interviewer is checking for.
3. **Data**: Game outcomes (win/loss/draw) between rated players — no
   labeling needed, the label *is* the game result, arriving immediately
   (no label-delay problem, unlike Q1/Q3).
4. **Features**: None in the ML-features sense — Elo needs only the two
   players' current ratings and the game outcome to update both.
5. **"Model"**: Elo update rule `R' = R + K(S - E)` where `E` is the
   expected score from the rating difference (Lesson 048's minimax bot
   would start at a fixed rating estimated from calibration games against
   known-strength opponents, e.g. Projects 008/009/010's three bot
   versions naturally forming a rating ladder).
6. **Offline eval**: Backtest the rating system on historical game logs
   (if any exist) — check that rating changes stabilize over time and
   correlate with actual win rate.
7. **Serving**: Matchmaking is a lookup/query problem (find opponents
   within a rating band of the requesting player) — a database query
   pattern you already know, not a model-serving problem at all.
8. **Online eval**: Monitor match quality proxies — win-rate spread
   staying close to 50/50 for matched pairs, queue wait times if the
   rating band is too narrow.
9. **Monitoring**: Rating drift for a bot after an engine upgrade (e.g.
   moving from Project 008's minimax to Project 010's MCTS+network bot) —
   its rating should be allowed to re-calibrate via provisional-rating
   mechanics (as real chess rating systems do for new/returning players).
10. **Tradeoff**: A wider matchmaking band finds opponents faster but
    produces less balanced games — the classic matchmaking latency/quality
    tradeoff, worth naming explicitly.

## 2. E-commerce recommendations — key points

Cold start for new users: fall back to popularity/trending items or
onboarding-survey-based cold-start features rather than collaborative
filtering, which needs interaction history. Cold start for new products:
use content-based features (category, text description via Lesson 056/057
embeddings) until enough interaction data accumulates to blend in
collaborative signals. Mention a two-stage architecture (cheap candidate
generation over the full catalog, then a heavier ranking model over the
narrowed candidate set) — the standard large-scale recsys pattern.

## 3. Content moderation — key points

Explicitly state the asymmetric cost: over-removal (false positive) harms
legitimate users and trust; under-removal (false negative) risks real
harm and platform liability — these usually aren't symmetric, and the
threshold choice (Lesson 025's precision/recall tradeoff) should be
justified against whichever asymmetry the business states. A common
pattern worth naming: three-tier action (auto-remove above a high-
confidence threshold, auto-flag-for-human-review in a middle band, allow
below a low threshold) rather than a single binary cutoff.

## 5. LLM customer support assistant — key points

RAG grounding (Lesson 076) to reduce hallucination on company-specific
facts (pricing, policies); explicit uncertainty handling — a confident-
but-wrong answer is worse than "I'm not sure, let me connect you to a
human," so design an explicit escalation trigger (low retrieval-
similarity score, or a small "can this be answered from retrieved
context" classifier) rather than always letting the LLM attempt an
answer. Mention the serving stack from Lessons 073–080 for latency/cost
at scale, and note that unlike Q1/Q3, evaluating a generative system
needs both automatic proxies (Lesson 073's benchmarks) and human review
of a sample, since there's no single ground-truth label per response.
