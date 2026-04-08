FROM ghcr.io/meta-pytorch/openenv-base:latest

LABEL maintainer="lexforge"
LABEL description="LexForge — Comprehensive Legal Intelligence OpenEnv Environment"
LABEL space_sdk="docker"

WORKDIR /app

# Install deps first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy server source
COPY . .

# Copy models.py from parent (needed by environment.py)
# In HF Space the full repo is copied; locally we symlink via PYTHONPATH
ENV PYTHONPATH=/app:/repo

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

EXPOSE 7860

ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]