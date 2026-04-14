# syntax=docker/dockerfile:1.6
FROM python:3.11-slim AS base

# Vars de ambiente
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependências de sistema mínimas + curl pro healthcheck
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Instala dependências Python (layer cacheada)
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Copia APENAS o código que a API precisa
COPY src/core ./src/core
COPY src/dl/models.py ./src/dl/models.py
COPY src/dl/__init__.py ./src/dl/__init__.py
COPY src/serving ./src/serving

# Copia o modelo treinado
COPY artifacts/model/lstm.pth ./artifacts/model/lstm.pth

# __init__.py vazios pra Python reconhecer os pacotes
RUN touch src/__init__.py src/dl/__init__.py

# Cria usuário não-root (boa prática de segurança)
RUN useradd --create-home --shell /bin/bash --uid 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Healthcheck: o Render vai usar /health pra saber se a API tá viva
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]