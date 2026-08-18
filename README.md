# Sentiment Classifier

Full-stack inference app for a DistilBERT sentiment model that was already fine-tuned in Google Colab.

This repository does **not** train models, download datasets, or run Hugging Face `Trainer` jobs. It only loads a local checkpoint, serves predictions, and (later) displays results in a Next.js UI.

```
Next.js Frontend
        ↓
FastAPI Backend
        ↓
Fine-tuned DistilBERT
        ↓
Sentiment Prediction
        ↓
Confidence + Probabilities
        ↓
Next.js UI
```

## Project layout

```
sentiment-classifier/
├── frontend/                 # Next.js UI (structure only for now)
├── backend/                  # FastAPI inference API
├── model/
│   └── sentiment-distilbert/ # Place the Colab-exported checkpoint here
├── README.md
├── .gitignore
└── docker-compose.yml
```

## Place the trained model

Export the fine-tuned Hugging Face checkpoint from Colab into:

```
model/sentiment-distilbert/
```

Typical files from `save_pretrained()`:

- `config.json`
- `tokenizer.json` / `tokenizer_config.json` / `vocab.txt`
- `model.safetensors` or `pytorch_model.bin`

Do not commit the weights. The directory is kept in git; the checkpoint files are ignored.

## Run locally

### Backend

The checkpoint location is configurable with `MODEL_PATH` (default
`model/sentiment-distilbert`, resolved relative to the repository root):

```bash
export MODEL_PATH=model/sentiment-distilbert   # optional
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API: `http://localhost:8000`  
Docs: `http://localhost:8000/docs`

### Frontend

```bash
cd frontend
npm install
npm run dev
```

UI: `http://localhost:3000`

Next.js App Router + TypeScript + Tailwind CSS v4, calling the FastAPI backend.

Point it at the API with `NEXT_PUBLIC_API_URL` (defaults to
`http://localhost:8000`):

```bash
cp .env.local.example .env.local
```

`NEXT_PUBLIC_*` values are inlined when the page is compiled, so change it
before `npm run build`. The Docker image runs `next dev`, which picks the value
up from the container environment at request time.

```
lib/api.ts                      predictSentiment() -> POST /predict
lib/types.ts                    PredictionRequest / PredictionResponse / Probabilities
lib/constants.ts                API_BASE_URL — the only place the URL is read
app/page.tsx                    layout only — header, subtitle, footer
components/SentimentAnalyzer    owns the idle/loading/success/error state
components/SentimentInput       textarea, character counter, actions
components/SentimentResult      hero confidence figure + breakdown
components/ConfidenceChart      per-class probability bars
components/ExampleReviews       one-click sample texts
components/LoadingState         skeleton while analyzing
components/ErrorState           inline failure panel
```

### Docker

```bash
docker compose up --build
```

Requires the checkpoint to already exist under `model/sentiment-distilbert/`.

## API

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Liveness check — `{"status": "ok"}` |
| POST | `/predict` | Classify one text |

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"text": "I really enjoyed this movie."}'
```

```json
{
  "text": "I really enjoyed this movie.",
  "sentiment": "positive",
  "confidence": 0.9971,
  "probabilities": {"negative": 0.0029, "positive": 0.9971}
}
```

Errors: `422` invalid or blank text, `500` inference failure, `503` model unavailable.

### Layering

```
api/routes.py       HTTP only — validate, delegate, shape the response
schemas.py          request/response contracts
services/sentiment_service.py  validate -> tokenize -> logits -> softmax -> label
models/model_loader.py  load the checkpoint once at startup
config.py           settings (MODEL_PATH, CORS origins)
```

## Tests

```bash
cd backend
pip install -r requirements-dev.txt
pytest                  # fast suite, fake model, no weights loaded
pytest -m integration   # runs the real checkpoint from MODEL_PATH
```

The default suite never loads, trains, or downloads a model: `tests/conftest.py`
supplies a fake tokenizer and classifier whose tensors flow through the real
service code, and the FastAPI dependency is swapped with
`app.dependency_overrides`. Integration tests skip themselves when the
checkpoint is missing.

## Status

| Area | Status |
|------|--------|
| Project structure | Ready |
| Backend config + health check | Ready |
| Model loading (startup, single instance) | Ready |
| Prediction inference | Ready |
| Prediction REST endpoint | Ready |
| Automated tests (pytest) | Ready |
| Frontend UI | Ready |
| Frontend ↔ API integration | Ready |
| Training / datasets | Out of scope |
# sentiment-classifier
