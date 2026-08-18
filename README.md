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

```bash
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

### Docker

```bash
docker compose up --build
```

Requires the checkpoint to already exist under `model/sentiment-distilbert/`.

## Status

| Area | Status |
|------|--------|
| Project structure | Ready |
| Backend config + health check | Ready |
| Model loading / inference | Not implemented |
| Prediction REST endpoint | Not implemented |
| Frontend UI | Not implemented |
| Training / datasets | Out of scope |
# sentiment-classifier
