# syntax=docker/dockerfile:1

# Inference-only image for the FastAPI backend.
#
# This image NEVER trains a model and NEVER downloads one. The fine-tuned
# checkpoint is supplied at runtime through a mounted volume (see
# docker-compose.yml) and located via MODEL_PATH.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # Hard offline switches: transformers cannot reach the Hugging Face Hub, so
    # a missing checkpoint fails loudly instead of pulling a replacement model.
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    # Default checkpoint location inside the container; override at runtime.
    MODEL_PATH=/model/sentiment-distilbert

WORKDIR /app

# CPU-only torch by default — the CUDA wheel pulls in several GB of NVIDIA
# runtime libraries. For a GPU image, build with:
#   --build-arg TORCH_INDEX_URL=https://pypi.org/simple
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

COPY requirements.txt ./

# torch first, from the chosen index; everything else from PyPI. The second
# install sees torch already satisfied and leaves it alone.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --index-url "${TORCH_INDEX_URL}" torch \
    && pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Run as a non-root user; it only ever reads the checkpoint.
RUN useradd --create-home --uid 1001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Fails until the model has finished loading, so dependants can wait on it.
HEALTHCHECK --interval=15s --timeout=5s --start-period=90s --retries=5 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
