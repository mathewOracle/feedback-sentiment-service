"""Tests for the FastAPI endpoints."""
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_index_page_loads():
    response = client.get("/")
    assert response.status_code == 200
    assert "feedback-sentiment-service" in response.text


def test_api_analyze_positive_text():
    response = client.post("/api/analyze", json={"text": "I love this, it works great!"})
    assert response.status_code == 200
    body = response.json()
    assert body["label"] == "positive"
    assert 0.0 <= body["confidence"] <= 1.0


def test_form_analyze_returns_html_fragment():
    response = client.post("/analyze", data={"text": "Terrible, it broke immediately."})
    assert response.status_code == 200
    assert "negative" in response.text.lower()
