# feedback-sentiment-service

A small, self-contained sentiment classification microservice: TF-IDF +
Logistic Regression under the hood, FastAPI + HTMX on top. Ships with a
synthetic, templated dataset so it trains and runs with zero external
downloads -- no HuggingFace weights, no API keys, no GPU required.

## Why this exists

"Classify this feedback as positive/negative/neutral" is one of the most
common small-ML asks in any product org (support tickets, reviews,
survey responses). This is the minimal, boring, reliable version of that
service -- a good starting point before reaching for a heavier
transformer-based model.

## Stack

FastAPI + scikit-learn + HTMX + Tailwind (CDN). Model trains in well
under a second on the bundled dataset (~450 synthetic examples).

## Quickstart

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt

# optional: generate dataset + train + save model explicitly
python -m app.train

uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000, type some feedback, hit "Analyze sentiment".

Or hit the JSON API directly:

```bash
curl -X POST http://127.0.0.1:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "The delivery was late but support fixed it quickly."}'
```

## Project layout

```
app/
  data/
    generate_reviews.py   # synthetic dataset generator (deterministic seed)
    sample_reviews.csv     # generated dataset, checked in for reproducibility
  model.py                 # pipeline definition, train/load/predict
  train.py                 # CLI: generate data (if missing) + train + evaluate + save
  main.py                  # FastAPI routes (JSON API + HTMX form endpoint)
templates/
  index.html                # page shell + form
  partial_result.html        # HTMX-swappable result fragment
tests/
  test_model.py
  test_api.py
```

## Design notes

- **No model file committed to git** -- `load_or_train` trains on the fly
  from the checked-in CSV if `models/sentiment_model.joblib` doesn't
  exist yet. Fresh clone, zero setup, no binary artifacts in version
  control.
- **Synthetic dataset, not scraped reviews** -- `generate_reviews.py`
  combines template sentences across generic product/service subjects
  with a fixed random seed, so the CSV is fully reproducible and contains
  no real customer or scraped data.
- **TF-IDF + Logistic Regression over a transformer** -- for a
  templated, moderately-sized dataset this trains instantly, needs no
  GPU, and is easy to explain (feature weights are inspectable). Swap in
  a transformer later if the problem actually calls for it -- YAGNI
  until then.

## Testing

```bash
uv pip install -r requirements-dev.txt
pytest
```

## License

MIT.
