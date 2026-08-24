"""Generate synthetic human-written vs AI-generated-style text. Stdlib only.

The stylistic patterns here (formal transitions, hedged uniformity for
"AI"; contractions, informal asides for "human") are a simplified stand-in
for real stylometric differences studied in actual AI-text-detection
research - good enough to practice a real classification pipeline on,
not a claim that real AI text always looks like this.
"""
import csv
import random
from pathlib import Path

random.seed(7)

TOPICS = ["climate change", "remote work", "electric cars", "social media", "artificial intelligence",
          "space exploration", "healthy eating", "online education", "renewable energy", "urban planning"]

AI_TEMPLATES = [
    "In conclusion, {topic} is a multifaceted issue that requires careful consideration of various factors.",
    "It is important to note that {topic} has both advantages and disadvantages worth examining.",
    "Furthermore, the impact of {topic} on society cannot be overstated, as it affects numerous stakeholders.",
    "Overall, {topic} represents a significant area of interest with implications for the future.",
    "When considering {topic}, one must take into account multiple perspectives and potential outcomes.",
    "Additionally, research suggests that {topic} plays a crucial role in shaping modern discourse.",
    "To summarize, {topic} is an evolving field that continues to attract considerable attention.",
    "It is worth noting that {topic} presents both opportunities and challenges for various communities.",
]

HUMAN_TEMPLATES = [
    "honestly {topic} is kind of a mess right now, not gonna lie",
    "so I've been reading about {topic} and idk, it's complicated tbh",
    "my take on {topic}? it's overhyped but whatever, still interesting",
    "ugh {topic} again, everyone's talking about it but nobody agrees",
    "not sure what to think about {topic}, my friend says it's a big deal",
    "{topic} is cool i guess but also kinda scary if you think about it",
    "lol {topic} is literally all over my feed rn, so annoying",
    "ok so {topic} - i have mixed feelings, some good some bad stuff there",
]


def fill(template):
    return template.format(topic=random.choice(TOPICS))


def generate(n=500):
    rows = []
    for _ in range(n):
        is_ai = random.random() < 0.5
        template = random.choice(AI_TEMPLATES if is_ai else HUMAN_TEMPLATES)
        text = fill(template)
        rows.append({"text": text, "label": 1 if is_ai else 0})
    random.shuffle(rows)
    return rows


if __name__ == "__main__":
    rows = generate(500)
    out_path = Path(__file__).parent / "human_vs_ai.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out_path}")
