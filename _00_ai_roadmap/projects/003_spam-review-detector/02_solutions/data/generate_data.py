"""Generate synthetic SMS-spam and fake-review datasets. Stdlib only.

Usage:
    python generate_data.py
    -> writes sms_spam.csv and fake_reviews.csv next to this script
"""
import csv
import random
from pathlib import Path

random.seed(42)

# --- SMS spam ---

SPAM_TEMPLATES = [
    "congratulations you have won a {prize} claim now",
    "free {prize} click this link to claim your prize now",
    "urgent your account will be suspended verify now",
    "you are a winner claim your {prize} today limited time",
    "act now to get {prize} for free no cost no catch",
    "final notice claim your free {prize} before it expires",
    "cash prize alert you have been selected winner claim now",
    "limited offer free {prize} text back to claim yours today",
    "your number has won {prize} reply now to redeem",
    "exclusive deal free {prize} only for you click here now",
]
PRIZES = ["cash", "iphone", "gift card", "vacation", "voucher", "lottery ticket", "prize"]

HAM_TEMPLATES = [
    "hey are we still on for {activity} today",
    "can you pick up {item} on your way home",
    "meeting moved to {time} tomorrow let me know if that works",
    "thanks for the help with the {item} yesterday",
    "just checking in how was your {activity} this weekend",
    "reminder the {activity} starts at {time}",
    "let's grab lunch sometime this week",
    "the report on {item} is due by {time} friday",
    "happy birthday hope you have a great {activity}",
    "running a bit late for {activity} be there in ten minutes",
    # deliberately ambiguous ham: reuses spam-flavored words in an innocuous context
    "thanks so much for the free tickets to the {activity} tonight",
    "the {activity} call is free tonight so feel free to join",
    "you won the office raffle claim your prize at the front desk",
    "final reminder the {activity} rsvp deadline is {time} today",
]
ACTIVITIES = ["dinner", "meeting", "game", "trip", "call", "party", "workout"]
ITEMS = ["groceries", "the project", "the documents", "milk", "the invoice"]
TIMES = ["3pm", "noon", "9am", "5pm", "10am"]


def fill(template):
    return template.format(
        prize=random.choice(PRIZES),
        activity=random.choice(ACTIVITIES),
        item=random.choice(ITEMS),
        time=random.choice(TIMES),
    )


def generate_sms(n=500):
    rows = []
    for _ in range(n):
        is_spam = random.random() < 0.35
        template = random.choice(SPAM_TEMPLATES if is_spam else HAM_TEMPLATES)
        text = fill(template)
        rows.append({"text": text, "label": 1 if is_spam else 0})
    random.shuffle(rows)
    return rows


# --- Fake reviews ---

FAKE_TEMPLATES = [
    "best product ever! ! ! five stars amazing perfect buy now",
    "amazing product highly recommend everyone should buy this now",
    "incredible quality changed my life buy it immediately five stars",
    "perfect item exceeded all expectations best purchase ever made",
    "wow just wow this product is life changing everyone needs this",
    "five stars amazing incredible perfect must buy right now today",
    "best purchase of my life amazing quality super fast shipping perfect",
    "this product is a miracle everyone should own one immediately",
]
GENUINE_TEMPLATES = [
    "the product works as described shipping took about {days} days",
    "decent quality for the price had a minor issue with the {part}",
    "does the job but the {part} feels a bit cheap overall okay purchase",
    "good value took a few days to arrive works fine so far",
    "the {part} broke after a few weeks otherwise it was fine",
    "solid product does what it says would consider buying again",
    "average experience the {part} could be better but it works",
    "arrived on time packaging was fine product matches the description",
    # deliberately enthusiastic-but-genuine: overlaps with fake-review vocabulary
    "amazing product genuinely happy with this purchase five stars from me",
    "best {part} replacement i've bought so far highly recommend it",
]
DAYS = ["three", "five", "seven", "two"]
PARTS = ["handle", "battery", "screen", "casing", "strap", "lid"]


def fill_review(template):
    return template.format(days=random.choice(DAYS), part=random.choice(PARTS))


def generate_reviews(n=500):
    rows = []
    for _ in range(n):
        is_fake = random.random() < 0.4
        template = random.choice(FAKE_TEMPLATES if is_fake else GENUINE_TEMPLATES)
        text = fill_review(template)
        rows.append({"text": text, "label": 1 if is_fake else 0})
    random.shuffle(rows)
    return rows


def write_csv(rows, filename):
    out_path = Path(__file__).parent / filename
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    write_csv(generate_sms(500), "sms_spam.csv")
    write_csv(generate_reviews(500), "fake_reviews.csv")
