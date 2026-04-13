import numpy as np
import pandas as pd
import pytest

from src.dl.datasets import (
    create_sequences,
    fit_feature_scaler,
    split_train_val_by_unit,
)

def test_split_no_unit_leakage(synthetic_cmapss_df):
    # Critico> unit_id nao deve aparecer em treino e em val ao mesmo tempo
    train_df, val_df = split_train_val_by_unit(synthetic_cmapss_df, val_ratio = 0.2)
    train_units = set(train_df["unit_id"].unique())
    val_units = set(val_df["unit_id"].unique())
    assert train_units.isdisjoint(val_units), "Unidades de treino e validação devem ser distintas"
    
def test_create_sequences_shape(synthetic_cmapss_df):
    X, y = create_sequences(synthetic_cmapss_df, seq_length=20)
    assert X.ndim == 3
    assert X.shape[1] == 20
    assert len(X) == len(y)
    
def test_scaler_no_target_leakage(synthetic_cmapss_df):
    # Critico> scaler deve ser fitado apenas com features, sem target
    _, feature_cols = fit_feature_scaler(synthetic_cmapss_df)
    forbidden = {"rul", "unit_id", "cycle"}
    assert not forbidden.intersection(feature_cols), \
        f"Features proibidas no scaler: {forbidden.intersection(feature_cols)}"