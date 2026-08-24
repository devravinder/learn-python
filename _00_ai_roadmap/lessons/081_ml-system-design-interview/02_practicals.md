# 02 — Practicals: ML System Design Interview Prep

For each prompt, write a structured answer following the 10-step
framework from `01_concepts.md`, time-boxed to 45 minutes as if in a real
interview. Bullet points are fine — the goal is practicing the
*structure* and catching yourself if you skip a step, not prose quality.

1. **Design a spam detection system for a messaging app.** (Directly
   related to Project 003 — use it as your data/feature starting point,
   then design the production system around it: labeling pipeline, drift
   from spammers adapting, real-time scoring latency budget.)

2. **Design a product recommendation system for an e-commerce site's
   homepage.** Address cold start (new users with no history, new
   products with no interactions) explicitly — a case the framework's
   step 3/4 doesn't automatically cover and interviewers specifically
   probe for.

3. **Design a content moderation system that flags policy-violating text
   posts.** Discuss the precision/recall tradeoff explicitly: what does a
   false positive cost (wrongly removing legitimate content) vs. a false
   negative (a violation staying up) — and how that tradeoff should shape
   your classification threshold (ties back to Lesson 025).

4. **Design the matchmaking/rating system for the Chess Bot from Projects
   008–010** if it were deployed as a public multiplayer service (rate
   opponents' relative skill, decide who plays whom). This isn't a
   classification/regression problem in the usual sense — practice
   framing an unusual problem into the same 10-step structure.

5. **Design an LLM-powered customer support assistant** (ties together
   Modules 13–14: RAG for company-specific knowledge, the serving/
   deployment stack from Lessons 073–080, and an escalation path to a
   human agent). Discuss what happens when the model is confidently
   wrong — a failure mode with no clean analogue in a typical CRUD system
   design question.

6. Pick **one** of the above and go one level deeper: sketch the actual
   data schema/API contract (request/response shapes) as if you were
   about to hand it to a team to implement — the same level of detail
   you'd give for a plain backend system design answer, applied here.
