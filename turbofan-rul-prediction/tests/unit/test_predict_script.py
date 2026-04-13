from __future__ import annotations

import sys

import joblib
import pandas as pd
from sklearn.dummy import DummyRegressor

import src.inference.predict as pred


def test_predict_script_writes_predictions(tmp_path, monkeypatch):
    model_path = tmp_path / "model.joblib"
    input_path = tmp_path / "test_features.parquet"
    output_path = tmp_path / "preds.parquet"

    # Modelo dummy (sempre prevê a média)
    reg = DummyRegressor(strategy="mean")
    reg.fit([[0.0], [1.0]], [10.0, 20.0])
    joblib.dump(reg, model_path)

    df = pd.DataFrame(
        {
            "unit_id": [1, 1],
            "cycle": [1, 2],
            "rul": [5, 4],
            "feat": [0.123, -0.456],
        }
    )
    df.to_parquet(input_path, index=False)

    monkeypatch.setattr(pred, "BEST_MODEL_FILE", model_path)
    monkeypatch.setattr(pred, "ensure_directories", lambda: None)

    monkeypatch.setattr(
        sys,
        "argv",
        ["prog", "--input", str(input_path), "--output", str(output_path)],
    )

    pred.main()

    out = pd.read_parquet(output_path)
    assert "predicted_rul" in out.columns
    assert len(out) == len(df)
