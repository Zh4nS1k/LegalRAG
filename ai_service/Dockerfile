FROM python:3.10-slim

WORKDIR /app

# Project-local model cache (avoids Permission denied: '/app'). Persist via volume.
ENV HF_HOME=/app/.models_cache \
    HF_HUB_CACHE=/app/.models_cache \
    TRANSFORMERS_CACHE=/app/.models_cache \
    SENTENCE_TRANSFORMERS_HOME=/app/.models_cache \
    LEGAL_RAG_HF_CACHE_DIR=/app/.models_cache

# Install system dependencies if needed (e.g., for certain python libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Add the extra index URL for PyTorch CPU
RUN pip install --no-cache-dir \
    -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Copy the application as a proper package
COPY . /app/ai_service

# Ensure Python can import from /app
ENV PYTHONPATH=/app

# Create writable model cache (avoids Permission denied)
RUN mkdir -p /app/.models_cache && chmod 777 /app/.models_cache

# Pre-download models at build time (enable with PREFETCH_HF_MODEL=1)
ARG PREFETCH_HF_MODEL=0
RUN if [ "$PREFETCH_HF_MODEL" = "1" ]; then \
      python -m ai_service.scripts.download_models; \
    fi

# Run the FastAPI app
CMD ["uvicorn", "ai_service.api.api:app", "--host", "0.0.0.0", "--port", "8000"]