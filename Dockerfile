FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TTS_JOB_DIR=/tmp/vizipvoice_tts_jobs

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        git \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-service.txt requirements-tts-runtime.txt pyproject.toml ./
RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install -r requirements-service.txt \
    && pip install -r requirements-tts-runtime.txt

COPY zipvoice ./zipvoice
COPY tts_service ./tts_service
COPY README.md LICENSE ./

EXPOSE 8000

CMD ["uvicorn", "tts_service.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
