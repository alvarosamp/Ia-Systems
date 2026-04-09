from __future__ import annotations

from fastapi import FastAPI, HTTPException

from src.serving.schema import SequenceRequest, PredictionResponse
from src.serving.inference import predict, metadata


app = FastAPI(
    title="Turbofan RUL Prediction API",
    description="Predict Remaining Useful Life using an optimized LSTM model.",
    version="1.0.0",
)


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
    try:
        result = predict(request.sequence, normalized=request.normalized)
        return {"predicted_rul": result}
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")