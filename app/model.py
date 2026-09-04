"""Sentiment model: TF-IDF + Logistic Regression. Small, fast, explainable --
no GPU, no external model downloads, trains in well under a second on the
bundled synthetic dataset.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

APP_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_PATH = APP_DIR / "data" / "sample_reviews.csv"
DEFAULT_MODEL_PATH = APP_DIR.parent / "models" / "sentiment_model.joblib"


def build_pipeline() -> Pipeline:
    """A fresh, untrained pipeline. Kept as its own function so tests and
    the training script share one definition of "what the model is".
    """
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, stop_words="english")),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )


def train_from_dataframe(df: pd.DataFrame) -> Pipeline:
    pipeline = build_pipeline()
    pipeline.fit(df["text"], df["label"])
    return pipeline


def load_dataset(path: str | Path = DEFAULT_DATASET_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def train_and_save(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    model_path: str | Path = DEFAULT_MODEL_PATH,
) -> Pipeline:
    df = load_dataset(dataset_path)
    pipeline = train_from_dataframe(df)
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    return pipeline


def load_or_train(model_path: str | Path = DEFAULT_MODEL_PATH) -> Pipeline:
    """Load a saved model if present, otherwise train one on the fly.

    This keeps the service runnable from a fresh clone with zero setup
    steps -- no binary model artifact needs to be committed to git.
    """
    model_path = Path(model_path)
    if model_path.exists():
        return joblib.load(model_path)
    return train_and_save(model_path=model_path)


def predict(pipeline: Pipeline, text: str) -> dict:
    """Predict a label + confidence for a single piece of text."""
    label = pipeline.predict([text])[0]
    probabilities = pipeline.predict_proba([text])[0]
    classes = pipeline.classes_
    confidence = float(max(probabilities))
    scores = {cls: round(float(prob), 4) for cls, prob in zip(classes, probabilities)}
    return {"label": label, "confidence": round(confidence, 4), "scores": scores}


def evaluate_holdout(dataset_path: str | Path = DEFAULT_DATASET_PATH, test_size: float = 0.2) -> float:
    """Train/test split accuracy check -- used by the training CLI to
    sanity-check the model isn't garbage before saving it.
    """
    df = load_dataset(dataset_path)
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=42, stratify=df["label"]
    )
    pipeline = train_from_dataframe(train_df)
    accuracy = pipeline.score(test_df["text"], test_df["label"])
    return float(accuracy)
