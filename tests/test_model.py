"""Tests for the sentiment model layer."""
import pandas as pd
import pytest

from app.data.generate_reviews import generate_rows
from app.model import build_pipeline, evaluate_holdout, predict, train_from_dataframe


@pytest.fixture(scope="module")
def trained_pipeline():
    rows = generate_rows(seed=42)
    df = pd.DataFrame(rows, columns=["text", "label"])
    return train_from_dataframe(df)


def test_build_pipeline_has_expected_steps():
    pipeline = build_pipeline()
    assert [name for name, _ in pipeline.steps] == ["tfidf", "clf"]


def test_predict_returns_expected_shape(trained_pipeline):
    result = predict(trained_pipeline, "I love this, it works perfectly!")
    assert result["label"] in {"positive", "negative", "neutral"}
    assert 0.0 <= result["confidence"] <= 1.0
    assert set(result["scores"].keys()) == {"positive", "negative", "neutral"}


def test_predict_clearly_positive_text(trained_pipeline):
    result = predict(trained_pipeline, "I absolutely love the product, it works great.")
    assert result["label"] == "positive"


def test_predict_clearly_negative_text(trained_pipeline):
    result = predict(trained_pipeline, "Terrible experience with the service, would not recommend.")
    assert result["label"] == "negative"


def test_holdout_accuracy_is_reasonably_high():
    accuracy = evaluate_holdout()
    # Synthetic, templated data should be easy to separate; this is a
    # sanity floor, not a rigorous ML benchmark.
    assert accuracy > 0.85
