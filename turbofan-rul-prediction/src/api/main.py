from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel


ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_FILE = ROOT_DIR / "artifacts" / "model" / "model.joblib"

app = FastAPI(title="Turbofan RUL Prediction API")


class PredictionRequest(BaseModel):
    features: dict[str, float]


def load_model() -> Any:
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {MODEL_FILE}")
    return joblib.load(MODEL_FILE)


model = None


@app.on_event("startup")
def startup_event():
    global model
    model = load_model()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: PredictionRequest):
    df = pd.DataFrame([payload.features])
    pred = model.predict(df)[0]
    return {"predicted_rul": float(pred)}