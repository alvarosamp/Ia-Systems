from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# =========================
# PATHS DO PROJETO
# =========================
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "model"

MODEL_FILE = ARTIFACTS_DIR / "model.joblib"

app = FastAPI(title="Turbofan RUL Prediction API")


# =========================
# SCHEMAS
# =========================
class PredictionFeatures(BaseModel):
    setting_1: float
    setting_2: float
    setting_3: float
    sensor_1: float
    sensor_2: float
    sensor_3: float
    sensor_4: float
    sensor_5: float
    sensor_6: float
    sensor_7: float
    sensor_8: float
    sensor_9: float
    sensor_10: float
    sensor_11: float
    sensor_12: float
    sensor_13: float
    sensor_14: float
    sensor_15: float
    sensor_16: float
    sensor_17: float
    sensor_18: float
    sensor_19: float
    sensor_20: float
    sensor_21: float


class PredictionRequest(BaseModel):
    features: PredictionFeatures


# =========================
# LOAD MODEL
# =========================
def load_model() -> Any:
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {MODEL_FILE}")
    return joblib.load(MODEL_FILE)


model = None


@app.on_event("startup")
def startup_event():
    global model
    model = load_model()


# =========================
# ENDPOINTS
# =========================
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_path": str(MODEL_FILE),
    }


@app.get("/features")
def get_expected_features():
    return {
        "expected_features": [
            "setting_1",
            "setting_2",
            "setting_3",
            "sensor_1",
            "sensor_2",
            "sensor_3",
            "sensor_4",
            "sensor_5",
            "sensor_6",
            "sensor_7",
            "sensor_8",
            "sensor_9",
            "sensor_10",
            "sensor_11",
            "sensor_12",
            "sensor_13",
            "sensor_14",
            "sensor_15",
            "sensor_16",
            "sensor_17",
            "sensor_18",
            "sensor_19",
            "sensor_20",
            "sensor_21",
        ]
    }


@app.post("/predict")
def predict(payload: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Modelo não foi carregado.")

    try:
        features_dict = payload.features.model_dump()
        df = pd.DataFrame([features_dict])

        pred = model.predict(df)[0]

        return {
            "predicted_rul": float(pred)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao gerar predição: {str(e)}")