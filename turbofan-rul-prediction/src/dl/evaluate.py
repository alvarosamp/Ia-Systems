from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.dl.dataset import create_sequences
from src.dl.models import LSTMModel


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT_DIR = Path(__file__).resolve().parents[2]
TEST_FILE = ROOT_DIR / "data" / "processed" / "test_features.parquet"
MODEL_FILE = ROOT_DIR / "artifacts" / "model" / "lstm.pth"


def evaluate_dl(model, X, y):
    model.eval()
    preds = []

    with torch.no_grad():
        for i in range(0, len(X), 256):
            batch = torch.tensor(X[i:i + 256], dtype=torch.float32).to(DEVICE)
            out = model(batch).cpu().numpy().squeeze()
            preds.extend(out)

    preds = np.array(preds)

    rmse = np.sqrt(mean_squared_error(y, preds))
    mae = mean_absolute_error(y, preds)
    r2 = r2_score(y, preds)

    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
    }


def main():
    seq_len = 51
    hidden_size = 230

    print(f"Using device: {DEVICE}")
    print(f"Loading test data from: {TEST_FILE}")
    print(f"Loading model from: {MODEL_FILE}")

    test_df = pd.read_parquet(TEST_FILE)
    X_test, y_test = create_sequences(test_df, seq_len=seq_len)

    model = LSTMModel(input_size=X_test.shape[2], hidden_size=hidden_size).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_FILE, map_location=DEVICE))

    metrics = evaluate_dl(model, X_test, y_test)

    print("\nDL evaluation metrics:")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()