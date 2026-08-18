# Model directory

This directory holds the fine-tuned sentiment classifier. **Nothing here is
built by this repository.**

## Where the model comes from

The model was fine-tuned **externally, in Google Colab**, from a pretrained
DistilBERT checkpoint. The training notebook is not part of this project.

**This application does not perform training.** The backend only loads the
finished checkpoint and runs inference on it. There is no training loop, no
dataset handling, no Hugging Face `Trainer`, and no download from the Hugging
Face Hub — the loader is called with `local_files_only=True`, so a missing or
incomplete checkpoint fails loudly instead of silently fetching a different
model from the internet.

## Where to put it

Extract the exported checkpoint into:

```
model/sentiment-distilbert/
```

A `save_pretrained()` export typically contains:

| File | Required | Purpose |
|------|----------|---------|
| `config.json` | yes | Architecture and the `id2label` mapping |
| `model.safetensors` *(or `pytorch_model.bin`)* | yes | The trained weights |
| `tokenizer.json` *(or `vocab.txt`)* | yes | Tokenizer vocabulary |
| `tokenizer_config.json` | recommended | Tokenizer settings |
| `training_args.bin` | no | Colab training arguments; unused at inference |

The checkpoint must declare the two expected labels, or startup fails:

```json
{ "id2label": { "0": "negative", "1": "positive" } }
```

## How the backend finds it

The backend reads the location from the **`MODEL_PATH`** environment variable:

```bash
MODEL_PATH=model/sentiment-distilbert
```

That is the default. A relative value resolves against the repository root, so
the API behaves the same wherever `uvicorn` is started from; an absolute path is
used as given (Docker Compose passes `/model/sentiment-distilbert`). Copy
`backend/.env.example` to `backend/.env` to override it.

The model is loaded **once**, at FastAPI startup, onto CUDA when available and
CPU otherwise, and put in `eval()` mode. Requests never trigger a load.

## Do not commit the weights

The checkpoint is far too large for git — `model.safetensors` alone is ~256 MB.
`.gitignore` excludes everything inside `model/sentiment-distilbert/` (plus
`*.safetensors`, `*.bin`, `*.pt` and friends anywhere in the tree); only the
empty directory is kept, via `.gitkeep`. A fresh clone therefore has **no
weights**, and the backend will refuse to start until you extract them here:

```
Model directory not found: .../model/sentiment-distilbert
```

Do not edit, re-save, or re-train the files in this directory. The application
treats them as read-only inputs.
