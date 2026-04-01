from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.dl.datasets import apply_feature_scaler, create_sequences
from src.dl.models_v2 import LSTMRegressor


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT_DIR = Path(__file__).resolve().parents[2]
TEST_FEATURES_FILE = ROOT_DIR / "data" / "processed" / "test_features.parquet"
MODEL_FILE = ROOT_DIR / "artifacts" / "model" / "lstm_v2.pth"


def compute_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"rmse": float(rmse), "mae": float(mae), "r2": float(r2)}


def main():
    checkpoint = torch.load(MODEL_FILE, map_location=DEVICE)

    test_df = pd.read_parquet(TEST_FEATURES_FILE)

    feature_cols = checkpoint["feature_cols"]

    class SimpleScaler:
        pass

    scaler = SimpleScaler()
    scaler.mean_ = np.array(checkpoint["scaler_mean"])
    scaler.scale_ = np.array(checkpoint["scaler_scale"])

    test_df = apply_feature_scaler(test_df, scaler, feature_cols)

    seq_len = checkpoint["seq_len"]
    hidden_size = checkpoint["hidden_size"]
    num_layers = checkpoint["num_layers"]
    dropout = checkpoint["dropout"]
    feature_dim = checkpoint["feature_dim"]

    X_test, y_test = create_sequences(test_df, seq_len=seq_len)

    model = LSTMRegressor(
        input_size=feature_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    preds = []

    with torch.no_grad():
        for i in range(0, len(X_test), 256):
            batch = torch.tensor(X_test[i:i + 256], dtype=torch.float32).to(DEVICE)
            out = model(batch).cpu().numpy()
            preds.extend(out)

    preds = np.array(preds).reshape(-1)

    metrics = compute_metrics(y_test, preds)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()