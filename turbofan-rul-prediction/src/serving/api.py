from __future__ import annotations

from fastapi import FastAPI, HTTPException
from src.serving.schema import SequenceRequest, PredictionResponse
from src.serving.inference import predict, checkpoint

app = FastAPI(
    title="Turbofan RUL Prediction API",
    description="Predict Remaining Useful Life using optimized LSTM model",
    version="1.0.0",
)


@app.get("/")
def root():
    return {"message": "API is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    return {
        "model": "LSTM v2",
        "seq_len": checkpoint["seq_len"],
        "feature_dim": checkpoint["feature_dim"],
        "hidden_size": checkpoint["hidden_size"],
        "num_layers": checkpoint["num_layers"],
        "dropout": checkpoint["dropout"],
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_rul(request: SequenceRequest):
    try:
        result = predict(request.sequence)
        return {"predicted_rul": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))