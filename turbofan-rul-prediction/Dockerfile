FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Dependências de sistema mínimas
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Instala deps Python
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Copia código
COPY src/core ./src/core
COPY src/dl/models.py ./src/dl/models.py
COPY src/dl/__init__.py ./src/dl/__init__.py
COPY src/serving ./src/serving

# Copia o modelo treinado
COPY artifacts/model/lstm.pth ./artifacts/model/lstm.pth

# __init__.py vazios pra Python achar os pacotes
RUN touch src/__init__.py

EXPOSE 8000

CMD ["uvicorn", "src.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]