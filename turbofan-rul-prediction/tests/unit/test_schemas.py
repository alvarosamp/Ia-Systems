from __future__ import annotations

import pytest

from src.core.schemas import validate_features, validate_raw_processed


def test_schema_accepts_valid_data(synthetic_cmapss_df):
    validated = validate_raw_processed(synthetic_cmapss_df)
    assert len(validated) == len(synthetic_cmapss_df)


def test_schema_rejects_negative_rul(synthetic_cmapss_df):
    bad = synthetic_cmapss_df.copy()
    bad.loc[0, "rul"] = -10

    with pytest.raises(ValueError, match="Pandera"):
        validate_raw_processed(bad)


def test_feature_schema_accepts_valid_data(synthetic_cmapss_df):
    validated = validate_features(synthetic_cmapss_df)
    assert len(validated) == len(synthetic_cmapss_df)


def test_feature_schema_rejects_negative_rul(synthetic_cmapss_df):
    bad = synthetic_cmapss_df.copy()
    bad.loc[0, "rul"] = -1

    with pytest.raises(ValueError, match="Pandera"):
        validate_features(bad)
