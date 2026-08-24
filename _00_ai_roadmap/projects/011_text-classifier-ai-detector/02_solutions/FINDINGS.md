# Findings — Human vs AI-Generated Text Detector

*(In-distribution accuracy and the generalization stress test below are
verified against an independent pure-Python Naive Bayes implementation
run directly against the generated CSV — not fabricated. `analysis.py`'s
sklearn-based TF-IDF/SVD numbers should be closely comparable; regenerate
the data and run it yourself to confirm.)*

## In-distribution accuracy is (again) near 100%

Same pattern as Project 003: **100/100 test accuracy** on a held-out split.
The AI-style templates (formal transitions: "in conclusion," "furthermore,"
"it is important to note") and human-style templates (informal markers:
contractions, "lol," "ngl," "tbh") use almost entirely non-overlapping
vocabulary, making this an easy classification problem by construction.

## Generalization is *better* here than Project 003's spam detector — and here's why

Testing on completely novel, hand-written sentences (not from any
template):

| Sentence | Predicted |
|---|---|
| "honestly not sure how i feel about all this ai stuff, kinda weird ngl" | human ✓ |
| "In summary, this topic warrants further investigation and analysis by researchers." | AI ✓ |
| "The weather today is nice." | **AI ✗ (should be human)** |
| "This is a great product I really enjoyed using it every day." | human ✓ |

3 of 4 generalize correctly — noticeably better than Project 003's spam
detector, which missed novel spam vocabulary entirely. The reason: this
classifier is really learning **register/formality**, a broad, transferable
stylistic signal, not a small fixed set of trigger words — so it
generalizes better to genuinely new sentences that clearly signal one
register or the other.

## But there's a real, systematic failure mode

**"The weather today is nice."** — a plain, neutral human sentence with
*no* informal markers (no slang, no contractions) — gets misclassified as
AI. This is not a random error: the classifier learned "informal markers
present -> human," so anything neutral or formal-but-human defaults toward
"AI," because the model never saw plain, neutral human writing during
training (every human example was deliberately casual). **This mirrors a
real, documented problem with actual AI-text detectors**: they
systematically produce false positives on writing that's simply formal,
technical, or written by a non-native English speaker who doesn't use much
slang — flagging real human writing as AI-generated purely because it
lacks informality markers. This is a genuine, reported harm of real
detection tools, not just a quirk of this toy dataset.

A related failure (not shown in the table): if an AI system is explicitly
instructed to "write casually," a style-based detector trained only on
*formal* AI examples would likely miss it entirely — the detector detects a
particular *style* of AI output, not "AI-ness" itself.

## Practical takeaway

Style-based detection generalizes better than pure keyword-spotting
(Project 003's spam case), but it's fundamentally still detecting **a
proxy** (formality/register) rather than the actual property of interest
(who/what wrote this) — and that proxy has known, systematic blind spots on
both ends (neutral human text flagged as AI; deliberately-casual AI text
missed entirely). This exact gap — needing a representation that captures
something closer to genuine authorship signal rather than surface style —
is part of the motivation for Project 012's fine-tuned Transformer
upgrade, though even state-of-the-art AI-text detectors today remain
imperfect and actively researched, not a solved problem.
