"""Builds a small, fully synthetic labeled dataset of customer-feedback-style
sentences for training the sentiment classifier. Deterministic (fixed seed)
so the checked-in CSV is reproducible from scratch with:

    python -m app.data.generate_reviews

No real customer data, no scraped reviews -- just template phrases combined
combinatorially across generic product/service nouns.
"""
from __future__ import annotations

import csv
import random
from pathlib import Path

SUBJECTS = [
    "the product", "the app", "this gadget", "the service", "the support team",
    "the delivery", "the packaging", "the software", "the subscription",
    "the checkout process", "the mobile experience", "the customer service",
    "this tool", "the update", "the new feature", "the website",
]

POSITIVE_TEMPLATES = [
    "I absolutely love {subject}, it works great.",
    "{subject_cap} exceeded my expectations, really impressed.",
    "Fantastic experience with {subject}, would recommend to anyone.",
    "{subject_cap} is fast, reliable, and easy to use.",
    "Great job on {subject}, everything worked perfectly.",
    "I'm very happy with {subject}, five stars.",
    "{subject_cap} made my day so much easier, thank you.",
    "Really smooth experience, {subject} is top notch.",
    "{subject_cap} works flawlessly and looks great too.",
    "Couldn't be happier with {subject}, will buy again.",
]

NEGATIVE_TEMPLATES = [
    "I'm really disappointed with {subject}, it barely works.",
    "{subject_cap} was a complete waste of money.",
    "Terrible experience with {subject}, would not recommend.",
    "{subject_cap} kept crashing and support was no help at all.",
    "Very frustrated with {subject}, this needs a lot of work.",
    "{subject_cap} arrived broken and customer service ignored me.",
    "Worst experience ever, {subject} does not work as advertised.",
    "{subject_cap} is slow, buggy, and confusing to use.",
    "I regret purchasing {subject}, total letdown.",
    "{subject_cap} failed within a week, extremely poor quality.",
]

NEUTRAL_TEMPLATES = [
    "{subject_cap} is okay, nothing special either way.",
    "I have mixed feelings about {subject}, some parts are fine.",
    "{subject_cap} does what it says, no complaints, no praise.",
    "It's an average experience with {subject}, could be better.",
    "{subject_cap} works as expected, pretty standard overall.",
    "Not bad, not great -- {subject} is just fine.",
    "{subject_cap} met the basic requirements, nothing more.",
    "I'm neutral on {subject}, it did the job I guess.",
    "{subject_cap} is fine for the price, average quality.",
    "Just an ordinary experience with {subject}, hard to say more.",
]

_TEMPLATES_BY_LABEL = {
    "positive": POSITIVE_TEMPLATES,
    "negative": NEGATIVE_TEMPLATES,
    "neutral": NEUTRAL_TEMPLATES,
}


def _render(template: str, subject: str) -> str:
    return template.format(subject=subject, subject_cap=subject[0].upper() + subject[1:])


def generate_rows(seed: int = 42) -> list[tuple[str, str]]:
    """Return a shuffled list of (text, label) pairs covering all subjects
    x all templates x all labels -- deterministic given the seed.
    """
    rng = random.Random(seed)
    rows: list[tuple[str, str]] = []
    for label, templates in _TEMPLATES_BY_LABEL.items():
        for subject in SUBJECTS:
            for template in templates:
                rows.append((_render(template, subject), label))
    rng.shuffle(rows)
    return rows


def write_csv(path: str | Path, seed: int = 42) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = generate_rows(seed=seed)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["text", "label"])
        writer.writerows(rows)
    return path


if __name__ == "__main__":
    out = write_csv(Path(__file__).resolve().parent / "sample_reviews.csv")
    print(f"Wrote synthetic dataset to {out}")
