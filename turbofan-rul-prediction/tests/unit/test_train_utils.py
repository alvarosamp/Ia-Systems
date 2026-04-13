from __future__ import annotations

import pandas as pd
import pytest
from omegaconf import OmegaConf

from src.training.train import build_model, split_Xy


def test_split_Xy_drops_standard_columns_and_returns_target():
    df = pd.DataFrame(
        {
            "unit_id": [1, 2],
            "cycle": [1, 1],
            "rul": [5, 6],
            "feat": [0.1, 0.2],
        }
    )

    X, y = split_Xy(df)

    assert list(X.columns) == ["feat"]
    assert y.tolist() == [5, 6]


def test_build_model_random_forest_returns_pipeline():
    cfg = OmegaConf.create(
        {
            "model": {"name": "random_forest", "params": {"n_estimators": 10}},
            "training": {"random_state": 42},
        }
    )

    model = build_model(cfg)

    # Pipeline com scaler + RandomForestRegressor
    assert hasattr(model, "fit")
    assert hasattr(model, "predict")
    assert getattr(model, "steps", None) is not None


def test_build_model_unknown_raises_value_error():
    cfg = OmegaConf.create(
        {
            "model": {"name": "does_not_exist", "params": {}},
            "training": {"random_state": 42},
        }
    )

    with pytest.raises(ValueError, match="Modelo nao suportado"):
        build_model(cfg)


def test_build_model_xgboost_if_available():
    pytest.importorskip("xgboost")

    cfg = OmegaConf.create(
        {
            "model": {"name": "xgboost", "params": {"n_estimators": 5, "max_depth": 2}},
            "training": {"random_state": 7},
        }
    )

    model = build_model(cfg)
    assert hasattr(model, "fit")
    assert hasattr(model, "predict")
