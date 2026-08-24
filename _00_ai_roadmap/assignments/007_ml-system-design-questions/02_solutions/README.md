# 02 — Model Answers: ML System Design Questions

Model answers, not "the" answer — graded on structure and reasoning
against the 10-step framework, not on matching this exactly.

## 1. Real-time fraud detection

1. **Clarify**: <200ms decision budget, extremely imbalanced classes
   (fraud is typically <1% of transactions), cost of a false negative
   (fraud approved) vs. false positive (legitimate transaction declined,
   angering a customer) are both real but usually asymmetric toward
   catching fraud, within a tolerable false-positive rate the business
   sets.
2. **Frame**: Binary classification, `P(fraud | transaction, account
   history)`, scored synchronously in the authorization path.
3. **Data/labels**: This is the crux of the question. Confirmed labels
   (chargebacks) arrive weeks later — training data is necessarily
   "stale" relative to the freshest transactions. Practical fix: train on
   a rolling window that ends a few weeks before "now" (accepting that
   the most recent weeks of data are unusable for training until their
   labels mature), and treat any transaction newer than the label-delay
   window as unlabeled for training purposes, not as a confirmed
   negative.
4. **Features**: Transaction amount vs. account's historical spending
   pattern, velocity features (transactions in the last hour/day),
   merchant category, device/location fingerprint vs. account's typical
   pattern. Must all be computable in the <200ms budget — anything
   requiring a slow join or external API call is disqualified regardless
   of predictive power.
5. **Model**: Gradient-boosted trees (Lesson 028) are the industry-
   standard baseline here — fast inference, handle tabular mixed
   features well, and are easier to audit/explain to a compliance team
   than a deep model, a real non-technical constraint worth naming.
6. **Offline eval**: PR-AUC, not accuracy (Lesson 025) — with such
   extreme class imbalance, a model predicting "never fraud" scores >99%
   accuracy while being useless.
7. **Serving**: Feature computation must be pre-aggregated/cached
   (velocity counts kept in a fast key-value store, not recomputed from
   a transaction log per request) to hit the latency budget — this is
   the feature-store pattern from Lesson 081 applied under a hard
   real-time constraint.
8. **Online eval**: Can't A/B test by literally letting fraud through for
   a control group — instead, shadow-deploy the new model (score
   transactions without acting on the score) and compare its decisions
   against the current production system's outcomes.
9. **Monitoring**: Fraud patterns adapt adversarially and quickly
   (fraudsters actively probe for what gets approved) — monitor
   precision/recall proxies more frequently than a typical model, and
   have a fast-retrain or rule-based override path for sudden new fraud
   patterns that predate the model noticing.
10. **Tradeoff**: A stricter threshold catches more fraud but declines
    more legitimate transactions — name the customer-trust cost
    explicitly, since it's real but doesn't show up in a PR-AUC number.

## 2. Job-listing search ranking

1. **Clarify**: Query could be a role title, skill, or company; personalize
   using the searcher's profile (past applications, skills, location).
2. **Frame**: Learning-to-rank, not classification — score each candidate
   job posting for this (query, user) pair and sort by score.
3. **Data**: Implicit labels from clicks/applications (a click is a weak
   positive, an application is a strong positive) — the same click-as-
   label pattern as Lesson 081's Q2, with the same self-selection bias
   caveat (you only observe clicks on postings that were already shown).
4. **Features**: Query-posting text match (Lesson 056/057), user-profile-
   to-posting-skill overlap, posting recency, employer reputation/
   response-rate signals.
5. **Model**: A two-stage architecture — cheap candidate retrieval
   (keyword/embedding search over the full posting index) followed by a
   learning-to-rank model (e.g. LambdaMART or a pairwise/listwise neural
   ranker) over the top few hundred candidates — the standard large-scale
   search pattern, reused from Lesson 081's e-commerce answer.
6. **Offline eval**: Ranking-specific metrics, not classification
   accuracy — NDCG or MRR against historical click/application logs,
   which weight getting the *top* results right much more heavily than
   getting the 50th result right, matching how users actually scan a
   results page.
7. **Serving**: Candidate retrieval must be fast (approximate nearest-
   neighbor search over embeddings, or an inverted index) since the full
   posting catalog can be large; only the narrowed candidate set goes
   through the heavier ranking model.
8. **Online eval**: A/B test on application rate and time-to-first-
   application, not just click-through rate — a ranking that maximizes
   clicks but not applications may be surfacing clickbait-y postings.
9. **Monitoring**: Job-market seasonality (hiring surges/freezes by
   industry) is a natural source of drift distinct from the adversarial
   drift in Q1 — monitor for posting-mix shifts, not just score
   distribution shifts.
10. **Tradeoff**: Optimizing purely for applications risks favoring
    postings with low application-effort (easy-apply) over better-fit
    but higher-effort postings — worth naming as a metric-gaming risk.

## 3. "What to watch next" recommendation

1. **Clarify**: Shown immediately after a video ends — the goal is
   session continuation (keep watching) more than long-term
   satisfaction, though both matter.
2. **Frame**: Ranking problem, same shape as Q2, but conditioned on a
   single, very recent, very strong signal — the just-watched video —
   rather than a broad user profile.
3. **Data**: Sequential watch-history data (this video was followed by
   that video, and the session continued vs. ended) — naturally suited
   to sequence modeling (Lesson 046/047's RNN/seq2seq content, or a
   Transformer over watch sequences) in a way the homepage case (Lesson
   081 Q2) isn't.
4. **Features**: The just-watched video's content embedding and
   category, co-watch statistics (videos frequently watched in sequence
   with this one), plus the user's longer-term profile as a secondary
   signal.
5. **Model**: A two-tower or sequence model that embeds "what was just
   watched" and retrieves similar/complementary videos — richer than the
   homepage's cold-start-heavy setting since there's always at least one
   strong, immediate signal available (the just-finished video), even for
   a brand-new user with no history.
6. **Offline eval**: Session-continuation rate (did the user watch the
   next video) and watch-time of the next video, evaluated against
   historical sessions.
7. **Serving**: Needs to be fast enough to show the "up next" panel with
   no perceptible delay as the current video ends — precompute
   candidate co-watch lists offline, rank live.
8. **Online eval**: A/B test on session length and total watch time, the
   platform's actual north-star metric, not just next-click rate.
9. **Monitoring**: Watch for feedback-loop effects — if the model always
   recommends similar content, session diversity can collapse over time,
   a failure mode worth monitoring even though it doesn't show up as a
   drop in short-term engagement metrics.
10. **Tradeoff**: Optimizing purely for session length can push toward
    increasingly narrow, addictive content loops — a real product/ethics
    tradeoff worth naming explicitly, distinct from a pure ML metric
    tradeoff.

## 4. Autocomplete/typeahead

1. **Clarify**: <50ms per keystroke, meaning almost everything must be
   precomputed — this is the most latency-constrained question in the
   set and should be framed as such immediately.
2. **Frame**: Given a prefix, rank candidate completions by predicted
   relevance (a mix of popularity and personalization) — not a
   classification problem.
3. **Data**: Historical query logs, aggregated by prefix — "given users
   typed 'mach', what did they end up searching most often" is
   essentially a groupby-and-count (Lesson 082, exercise 8) at massive
   scale.
4. **Features**: Prefix-to-completion frequency, personalization
   (user's own search history), trending/recency boost for suddenly
   popular queries.
5. **Model**: For the latency budget, this is closer to a precomputed
   lookup table (a trie or prefix index mapping prefix → top-N ranked
   completions, refreshed periodically) than a per-keystroke model
   inference call — a case worth explicitly naming as "the right answer
   here is mostly a data structure, not a model," mirroring the chess-
   matchmaking realization from Lesson 081 Q4.
6. **Offline eval**: How often the eventually-submitted query appeared
   in the top-N suggestions shown at some point during typing.
7. **Serving**: A precomputed prefix trie served from an in-memory
   store (Redis or equivalent) — the live "model" call is a fast lookup,
   with any personalization applied as a lightweight re-ranking on top of
   the precomputed candidates, keeping the hot path within budget.
8. **Online eval**: A/B test on suggestion-click rate and time-to-submit-
   query.
9. **Monitoring**: Trending-query detection needs a much shorter refresh
   cycle than the base popularity index (a breaking-news query should
   surface within minutes, not after the next scheduled batch job).
10. **Tradeoff**: Precomputing everything is fast but can't react
    instantly to brand-new trending queries — a hybrid (precomputed base
    + a fast trending-boost signal layered on top) is the practical
    answer, and naming this hybrid explicitly is stronger than picking
    one extreme.

## 5. Infrastructure anomaly detection / alerting

1. **Clarify**: Thousands of hosts, multiple metrics per host, alerts
   page a human — the cost of a false positive (paging someone at 3am
   for nothing) is concrete and immediate, unlike most classification
   problems where a false positive is more abstract.
2. **Frame**: Anomaly detection (unsupervised or semi-supervised) rather
   than classification — there's usually no labeled "this was an
   incident" dataset large enough to train a supervised classifier on
   directly, especially for novel failure modes.
3. **Data**: Time-series metrics per host; any past incident tickets
   provide sparse, valuable labels for validation even if not for direct
   supervised training.
4. **Features**: Per-metric time series, ideally decomposed into
   trend/seasonality (daily/weekly traffic patterns) so that "3x normal
   CPU at 2pm on a Tuesday" and "normal CPU for 2am" aren't both flagged
   as anomalies just because they differ from the 24-hour average.
5. **Model**: Start with a simple statistical baseline (e.g. seasonal
   decomposition + a threshold on the residual, similar in spirit to
   Lesson 007's forecasting baselines) before reaching for a learned
   anomaly detector — same "justify the complexity" discipline as
   Lesson 081's step 5, doubly important here since a simple, explainable
   threshold is much easier for an on-call engineer to trust than an
   opaque model score at 3am.
6. **Offline eval**: Backtest against past incident tickets — did the
   detector's anomaly score spike before/during each known past
   incident, and how many "anomalies" did it flag that correspond to
   nothing in the ticket history (a direct proxy for false-positive
   rate).
7. **Serving**: Needs to run continuously as a streaming computation
   over incoming metrics, not a request/response API — architecturally
   closer to a Node/Kafka streaming consumer you may have built than to
   Lesson 078's REST-endpoint pattern.
8. **Online eval**: Track the page-to-actual-incident ratio after
   deployment — this is the single most important number for combating
   alert fatigue, more important than any offline precision/recall
   figure.
9. **Monitoring**: The system needs meta-monitoring of itself — if
   alert volume spikes, is that a real infrastructure-wide incident or
   the anomaly detector itself misbehaving (e.g. after a legitimate
   traffic pattern change it hasn't adapted to yet)?
10. **Tradeoff**: This is the sharpest precision/recall tradeoff in the
    set — state explicitly that alert fatigue makes precision the
    priority metric here even at some cost to recall, the opposite
    emphasis from Q1's fraud system, and say so explicitly rather than
    defaulting to "maximize recall" as if every ML system shared the
    same priority.
