"""CLI entry point: generate dataset (if missing) + train + evaluate + save.

Run: python -m app.train
"""
from __future__ import annotations

from app.data.generate_reviews import write_csv
from app.model import DEFAULT_DATASET_PATH, DEFAULT_MODEL_PATH, evaluate_holdout, train_and_save


def main() -> None:
    if not DEFAULT_DATASET_PATH.exists():
        write_csv(DEFAULT_DATASET_PATH)
        print(f"Generated synthetic dataset at {DEFAULT_DATASET_PATH}")

    accuracy = evaluate_holdout(DEFAULT_DATASET_PATH)
    print(f"Holdout accuracy: {accuracy:.2%}")

    train_and_save(DEFAULT_DATASET_PATH, DEFAULT_MODEL_PATH)
    print(f"Trained model saved to {DEFAULT_MODEL_PATH}")


if __name__ == "__main__":
    main()
