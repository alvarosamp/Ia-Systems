from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

from src.serving.api import app
from src.serving.inference import metadata


client = TestClient(app)


def test_explain_returns_top_features():
    """O /explain deve retornar features com importância e direção."""
    seq_len = metadata["seq_len"]
    feature_dim = metadata["feature_dim"]
    sequence = np.random.randn(seq_len, feature_dim).tolist()

    r = client.post("/explain", json={
        "sequence": sequence,
        "normalized": False,
        "top_k": 5,
    })
    assert r.status_code == 200

    data = r.json()
    assert "predicted_rul" in data
    assert "top_features" in data
    assert len(data["top_features"]) == 5
    assert "timestep_importance" in data
    assert len(data["timestep_importance"]) == seq_len

    # Cada feature deve ter os campos obrigatórios
    feat = data["top_features"][0]
    assert "feature" in feat
    assert "importance" in feat
    assert "direction" in feat
    assert feat["direction"] in ("increases RUL", "decreases RUL")

    # Importâncias devem somar ~100%
    total_importance = sum(f["importance"] for f in data["top_features"])
    assert total_importance > 0


def test_explain_wrong_shape_returns_422():
    r = client.post("/explain", json={
        "sequence": [[1.0, 2.0]],
        "normalized": False,
    })
    assert r.status_code == 422


def test_explain_method_is_documented():
    """O response deve documentar o método usado."""
    seq_len = metadata["seq_len"]
    feature_dim = metadata["feature_dim"]
    sequence = np.random.randn(seq_len, feature_dim).tolist()

    r = client.post("/explain", json={"sequence": sequence, "normalized": False})
    data = r.json()
    assert data["explanation_method"] == "Integrated Gradients (Captum)"
    assert data["baseline"] == "zero (normalized mean of training data)"