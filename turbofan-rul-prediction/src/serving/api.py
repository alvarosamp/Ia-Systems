from __future__ import annotations

import time

from fastapi import FastAPI, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator

from src.serving.schema import (
    SequenceRequest,
    PredictionResponse,
    ExplainRequest,
    ExplainResponse,
)
from src.serving.inference import predict, metadata
from src.serving.explainer import explain_prediction
from src.monitoring.metrics import (
    PREDICTION_COUNT,
    PREDICTION_LATENCY,
    PREDICTED_RUL_DISTRIBUTION,
    MODEL_INFO,
)


app = FastAPI(
    title="Turbofan RUL Prediction API",
    description=(
        "Predict Remaining Useful Life of turbofan engines using an optimized LSTM model. "
        "Includes explainability via Integrated Gradients."
    ),
    version="2.0.0",
)

Instrumentator().instrument(app).expose(app)


@app.on_event("startup")
def _set_model_gauge():
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


@app.post("/explain", response_model=ExplainResponse)
def explain_rul(request: ExplainRequest):
    """
    Explica uma predição de RUL mostrando quais features e timesteps
    mais influenciaram o resultado.

    Usa Integrated Gradients (Captum) com baseline zero (= média do treino).
    Retorna as top-K features com importância relativa e direção do efeito.
    """
    try:
        result = explain_prediction(
            sequence=request.sequence,
            normalized=request.normalized,
            top_k=request.top_k,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explain error: {e}")