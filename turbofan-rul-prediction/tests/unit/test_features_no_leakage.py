import pandas as pd
import pytest 
from src.features.build_features import add_cycle_features, add_temporal_features

FORBIDDEN_FEATURES = {"cycle_ratio", "cycle_left_proxy", "cycle_squared"}

def test_cycle_features_have_no_leakage(synthetic_cmapss_df):
    """Regressão: cycle_ratio, cycle_left_proxy e cycle_squared são leakage."""
    result = add_cycle_features(synthetic_cmapss_df)
    leaked = FORBIDDEN_FEATURES.intersection(result.columns)
    assert not leaked, (
        f"Features de leakage reintroduzidas: {leaked}. "
        "Leia o comentário em add_cycle_features() antes de adicionar features de ciclo."
    )


def test_temporal_features_are_added(synthetic_cmapss_df):
    result = add_temporal_features(synthetic_cmapss_df)
    assert "sensor_1_rolling_mean_5" in result.columns
    assert "sensor_1_diff_1" in result.columns