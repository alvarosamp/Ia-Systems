from __future__ import annotations

import pandas as pd

from src.data.prepare import (
    RUL_CAP,
    build_column_names,
    clean_dataframe,
    clip_rul,
    compute_test_rul,
    compute_train_rul,
)


def test_build_column_names_shape_and_keys():
    cols = build_column_names()

    assert cols[0] == "unit_id"
    assert cols[1] == "cycle"

    assert "setting_1" in cols
    assert "setting_3" in cols

    assert "sensor_1" in cols
    assert "sensor_21" in cols

    assert len(cols) == 2 + 3 + 21


def test_compute_train_rul_matches_max_cycle_minus_cycle():
    df = pd.DataFrame(
        {
            "unit_id": [1, 1, 1, 2, 2],
            "cycle": [1, 2, 3, 1, 2],
        }
    )

    out = compute_train_rul(df)

    expected = [2, 1, 0, 1, 0]
    assert out["rul"].tolist() == expected


def test_compute_test_rul_adds_extra_rul_from_rul_file():
    test_df = pd.DataFrame(
        {
            "unit_id": [1, 1, 1, 2, 2],
            "cycle": [1, 2, 3, 1, 2],
        }
    )
    rul_df = pd.DataFrame([10, 20])

    out = compute_test_rul(test_df, rul_df)

    # unit 1 max_cycle=3 => rul = (3-cycle) + 10
    # unit 2 max_cycle=2 => rul = (2-cycle) + 20
    expected = [12, 11, 10, 21, 20]
    assert out["rul"].tolist() == expected


def test_clip_rul_caps_upper_tail():
    df = pd.DataFrame({"rul": [0, RUL_CAP + 10]})

    out = clip_rul(df, cap=RUL_CAP)

    assert out["rul"].tolist() == [0, RUL_CAP]


def test_clean_dataframe_deduplicates_and_sorts():
    df = pd.DataFrame(
        {
            "unit_id": [2, 1, 1, 1],
            "cycle": [1, 2, 1, 1],
            "rul": [0, 0, 0, 0],
        }
    )

    out = clean_dataframe(df)

    assert out.duplicated().sum() == 0
    assert out[["unit_id", "cycle"]].values.tolist() == [[1, 1], [1, 2], [2, 1]]
