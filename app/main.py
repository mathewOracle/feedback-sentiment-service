"""FastAPI app: serves both a JSON API and an HTMX-powered demo UI for the
sentiment classifier. Model is trained lazily on first use if no saved
model artifact exists yet (see app.model.load_or_train).
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from app.model import load_or_train, predict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

app = FastAPI(title="feedback-sentiment-service")
templates = Jinja2Templates(directory=str(PROJECT_ROOT / "templates"))

_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = load_or_train()
    return _pipeline


class AnalyzeRequest(BaseModel):
    text: str


class AnalyzeResponse(BaseModel):
    label: str
    confidence: float
    scores: dict[str, float]


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {"result": None, "submitted_text": ""})


@app.post("/analyze", response_class=HTMLResponse)
def analyze_form(request: Request, text: str = Form(...)):
    """HTMX-facing endpoint: returns an HTML fragment with the result."""
    result = predict(get_pipeline(), text) if text.strip() else None
    return templates.TemplateResponse(
        request,
        "partial_result.html",
        {"result": result, "submitted_text": text},
    )


@app.post("/api/analyze", response_model=AnalyzeResponse)
def analyze_api(payload: AnalyzeRequest):
    """JSON API for programmatic use (curl, other services, tests)."""
    result = predict(get_pipeline(), payload.text)
    return AnalyzeResponse(**result)


@app.get("/api/health")
def health():
    return {"status": "ok"}
