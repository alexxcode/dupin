# Dupin serving — multi-stage, slim, usuario no-root.

FROM python:3.11-slim AS builder
ENV PIP_NO_CACHE_DIR=1
WORKDIR /install
COPY serving/requirements.txt .
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install -r requirements.txt

FROM python:3.11-slim
ENV PATH="/opt/venv/bin:$PATH" PYTHONUNBUFFERED=1 PORT=8080
# LightGBM requiere OpenMP (libgomp1) en runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
# Solo el código que el serving importa (paridad: el mismo features/).
COPY data ./data
COPY features ./features
COPY evaluation ./evaluation
COPY training ./training
COPY serving ./serving

RUN useradd -m -u 1000 dupin && chown -R dupin:dupin /app
USER dupin
EXPOSE 8080

# El bundle se descarga al arrancar desde DUPIN_BUNDLE_URI (gs://...).
CMD ["sh", "-c", "uvicorn serving.app:app --host 0.0.0.0 --port ${PORT:-8080}"]
