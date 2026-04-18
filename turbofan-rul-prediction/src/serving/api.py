from __future__ import annotations

import time

from fastapi import FastAPI, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator

from src.serving.schema import SequenceRequest, PredictionResponse
from src.serving.inference import predict, metadata
from src.monitoring.metrics import (
    PREDICTION_COUNT,
    PREDICTION_LATENCY,
    PREDICTED_RUL_DISTRIBUTION,
    MODEL_INFO,
)


app = FastAPI(
    title="Turbofan RUL Prediction API",
    description="Predict Remaining Useful Life using an optimized LSTM model.",
    version="1.0.0",
)

# Auto-instrumenta todos os endpoints com métricas HTTP padrão
# (request_count, request_latency, request_size, response_size)
# Expõe /metrics pra Prometheus scrape
Instrumentator().instrument(app).expose(app)


@app.on_event("startup")
def _set_model_gauge():
    """Registra metadados do modelo como gauge no startup."""
    MODEL_INFO.labels(
        seq_len=str(metadata["seq_len"]),
        hidden_size=str(metadata["hidden_size"]),
        num_layers=str(metadata["num_layers"]),
    ).set(1)


@app.get("/")
def root():
    return {"message": "Turbofan RUL API is running"}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True}


@app.get("/model-info")
def model_info():
    return {
        "model": "LSTM Regressor",
        "seq_len": metadata["seq_len"],
        "feature_dim": metadata["feature_dim"],
        "hidden_size": metadata["hidden_size"],
        "num_layers": metadata["num_layers"],
        "dropout": metadata["dropout"],
        "feature_cols": metadata["feature_cols"],
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_rul(request: SequenceRequest):
    start = time.perf_counter()
    try:
        result = predict(request.sequence, normalized=request.normalized)

        # Métricas custom
        elapsed = time.perf_counter() - start
        PREDICTION_LATENCY.observe(elapsed)
        PREDICTION_COUNT.labels(status="success").inc()
        PREDICTED_RUL_DISTRIBUTION.observe(result)

        return {"predicted_rul": result}
    except ValueError as e:
        PREDICTION_COUNT.labels(status="error").inc()
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        PREDICTION_COUNT.labels(status="error").inc()
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")