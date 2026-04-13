from __future__ import annotations

import pandas as pd

from src.features.build_features import (
    add_cycle_features,
    align_columns,
    remove_constant_columns,
)


def test_remove_constant_columns_drops_constant_features():
    df = pd.DataFrame(
        {
            "unit_id": [1, 2],
            "cycle": [1, 1],
            "rul": [5, 6],
            "const": [1, 1],
            "var": [0.1, 0.2],
        }
    )

    out = remove_constant_columns(df)

    assert "const" not in out.columns
    assert "var" in out.columns


def test_align_columns_removes_constants_and_preserves_targets():
    train_df = pd.DataFrame(
        {
            "unit_id": [1, 1, 2],
            "cycle": [1, 2, 1],
            "rul": [2, 1, 0],
            "const": [7, 7, 7],
            "feat_a": [0.0, 1.0, 2.0],
        }
    )
    test_df = pd.DataFrame(
        {
            "unit_id": [3, 3],
            "cycle": [1, 2],
            "rul": [9, 8],
            "const": [7, 7],
            "feat_a": [3.0, 4.0],
        }
    )

    out_train, out_test = align_columns(train_df, test_df)

    assert "const" not in out_train.columns
    assert "const" not in out_test.columns

    assert out_train["rul"].tolist() == [2, 1, 0]
    assert out_test["rul"].tolist() == [9, 8]

    # Ambos devem ter o mesmo conjunto de colunas (ordem pode variar, mas aqui é determinística)
    assert out_train.columns.tolist() == out_test.columns.tolist()


def test_add_cycle_features_adds_cycle_norm_only():
    df = pd.DataFrame({"unit_id": [1, 1], "cycle": [10, 20], "rul": [5, 4]})

    out = add_cycle_features(df)

    assert "cycle_norm" in out.columns
    assert out["cycle_norm"].tolist() == [10 / 300.0, 20 / 300.0]
