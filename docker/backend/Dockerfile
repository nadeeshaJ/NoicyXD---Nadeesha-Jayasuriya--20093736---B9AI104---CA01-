FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./requirements-ml.txt
COPY sound_analytics_platform/backend/requirements.txt ./sound_analytics_platform/backend/requirements.txt

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements-ml.txt \
    && pip install --no-cache-dir -r sound_analytics_platform/backend/requirements.txt

COPY config/ ./config/
COPY src/ ./src/
COPY data/splits/ ./data/splits/
COPY reports/ ./reports/
COPY experiments/ ./experiments/
COPY sound_analytics_platform/backend/ ./sound_analytics_platform/backend/

WORKDIR /app/sound_analytics_platform/backend

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=5 \
    CMD curl -fsS http://127.0.0.1:8000/api/health | grep -q '"status":"ok"' || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
