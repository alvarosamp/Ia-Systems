import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.serving.api import app
from src.serving.inference import metadata


client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_model_info():
    r = client.get("/model-info")
    assert r.status_code == 200
    data = r.json()
    assert data["model"] == "LSTM Regressor"
    assert data["seq_len"] > 0


def test_predict_valid_sequence():
    seq_len = metadata["seq_len"]
    feature_dim = metadata["feature_dim"]
    sequence = np.random.randn(seq_len, feature_dim).tolist()

    r = client.post("/predict", json={"sequence": sequence, "normalized": False})
    assert r.status_code == 200
    assert "predicted_rul" in r.json()


def test_predict_wrong_shape_returns_422():
    r = client.post("/predict", json={"sequence": [[1.0, 2.0]], "normalized": False})
    assert r.status_code == 422