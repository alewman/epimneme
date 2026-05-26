FROM python:3.12-slim AS base

LABEL org.opencontainers.image.title="engram"
LABEL org.opencontainers.image.description="Persistent memory service for AI coding agents (PostgreSQL + pgvector)"
LABEL org.opencontainers.image.licenses="Apache-2.0"
LABEL org.opencontainers.image.source="https://github.com/alewman/engram"
LABEL org.opencontainers.image.url="https://github.com/alewman/engram"

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies for psycopg (libpq) and building native extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer caching)
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch && \
    pip install --no-cache-dir .

# Pre-download default embedding model so it's baked into the image
# This avoids a slow first-request download at runtime
# EPIMNEME_EMBEDDING_MODEL can override at runtime (e.g. BAAI/bge-large-en-v1.5)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" && \
    python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-large-en-v1.5')"

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

# Run with uvicorn
CMD ["uvicorn", "engram.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4", "--log-level", "info"]
