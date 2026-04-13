from __future__ import annotations

from src.training.evaluate import evaluate_regression


def test_evaluate_regression_perfect_prediction():
    y_true = [1.0, 2.0, 3.0]
    y_pred = [1.0, 2.0, 3.0]

    metrics = evaluate_regression(y_true, y_pred)

    assert metrics["rmse"] == 0.0
    assert metrics["mae"] == 0.0
    assert metrics["r2"] == 1.0
