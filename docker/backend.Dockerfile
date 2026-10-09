# Context is the PROJECT ROOT (set in docker-compose.yml), so COPY paths start there
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app/backend

# CPU-only torch first: the default build pulls ~2 GB of CUDA libraries
COPY backend/requirements.txt .
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

COPY backend/ /app/backend/
COPY documents/ /app/documents/

# Download the embedding model and build the vector index INTO the image.
# Edit a document -> `docker compose build backend` rebuilds the index.
RUN python document_loader.py

# From here on, never call the Hugging Face Hub at runtime
ENV HF_HUB_OFFLINE=1 \
    APP_ENV=production

EXPOSE 8000

# --forwarded-allow-ips: nginx (another container) sends the real client IP,
# which the rate limiter uses. The backend port is NOT published to the host.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--proxy-headers", "--forwarded-allow-ips=*"]
