from __future__ import annotations

import pandas as pd

from src.core.schemas import validate_features
from src.core.settings import (
    TRAIN_PROCESSED_FILE,
    TEST_PROCESSED_FILE,
    FEATURE_TRAIN_FILE,
    FEATURE_TEST_FILE,
    ensure_directories,
)


SENSOR_COLUMNS = [f"sensor_{i}" for i in range(1, 22)]
SETTING_COLUMNS = [f"setting_{i}" for i in range(1, 4)]


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features temporais por unit_id: rolling stats e diferenças."""
    df = df.copy()

    for col in SENSOR_COLUMNS:
        grouped = df.groupby("unit_id")[col]

        df[f"{col}_rolling_mean_5"] = (
            grouped.rolling(window=5, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        df[f"{col}_rolling_std_5"] = (
            grouped.rolling(window=5, min_periods=1)
            .std()
            .reset_index(level=0, drop=True)
            .fillna(0)
        )
        df[f"{col}_rolling_min_5"] = (
            grouped.rolling(window=5, min_periods=1)
            .min()
            .reset_index(level=0, drop=True)
        )
        df[f"{col}_rolling_max_5"] = (
            grouped.rolling(window=5, min_periods=1)
            .max()
            .reset_index(level=0, drop=True)
        )

        df[f"{col}_diff_1"] = grouped.diff().fillna(0)
        df[f"{col}_diff_3"] = grouped.diff(3).fillna(0)

    return df


def add_cycle_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features derivadas do ciclo atual.

    IMPORTANTE — features PROIBIDAS e por quê:

    - `cycle / max(cycle)` (cycle_ratio): leakage catastrófico.
      No train, max(cycle) é o ciclo da falha, então cycle_ratio é
      basicamente `1 - rul/max_cycle`. No test, max(cycle) é apenas o
      último ciclo observado (antes da falha), então a feature tem
      semântica completamente diferente → o modelo colapsa no test.

    - `max(cycle) - cycle` (cycle_left_proxy): é literalmente o RUL
      no train (por definição do target). Leakage direto.

    Usamos apenas uma normalização por constante fixa, que tem a mesma
    semântica em train e test.
    """
    df = df.copy()
    df["cycle_norm"] = df["cycle"] / 300.0
    return df


def remove_constant_columns(df: pd.DataFrame) -> pd.DataFrame:
    nunique = df.nunique(dropna=False)
    constant_cols = nunique[nunique <= 1].index.tolist()
    return df.drop(columns=constant_cols)


def align_columns(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Garante que train e test tenham exatamente as mesmas colunas de feature."""
    train_target = train_df["rul"]
    test_target = test_df["rul"]

    train_x = train_df.drop(columns=["rul"])
    test_x = test_df.drop(columns=["rul"])

    combined = pd.concat([train_x, test_x], axis=0)
    combined = remove_constant_columns(combined)

    final_columns = combined.columns.tolist()

    train_df = train_df[[c for c in final_columns if c in train_df.columns]].copy()
    test_df = test_df[[c for c in final_columns if c in test_df.columns]].copy()

    train_df["rul"] = train_target.values
    test_df["rul"] = test_target.values

    return train_df, test_df


def main() -> None:
    ensure_directories()

    train_df = pd.read_parquet(TRAIN_PROCESSED_FILE)
    test_df = pd.read_parquet(TEST_PROCESSED_FILE)

    train_df = add_temporal_features(train_df)
    test_df = add_temporal_features(test_df)

    train_df = add_cycle_features(train_df)
    test_df = add_cycle_features(test_df)

    train_df, test_df = align_columns(train_df, test_df)

    # Validação Pandera
    train_df = validate_features(train_df, name="train_features")
    test_df = validate_features(test_df, name="test_features")

    # Sanity check: nenhuma feature proibida deve ter sobrado
    forbidden = {"cycle_ratio", "cycle_left_proxy", "cycle_squared"}
    leaked = forbidden.intersection(train_df.columns)
    if leaked:
        raise RuntimeError(
            f"Features de leakage detectadas: {leaked}. "
            "Revise add_cycle_features() em build_features.py."
        )

    train_df.to_parquet(FEATURE_TRAIN_FILE, index=False)
    test_df.to_parquet(FEATURE_TEST_FILE, index=False)

    print(f"Train features salvas em: {FEATURE_TRAIN_FILE}")
    print(f"Test features salvas em:  {FEATURE_TEST_FILE}")
    print(f"Shape train: {train_df.shape}")
    print(f"Shape test:  {test_df.shape}")
    print(f"N features (sem unit_id/cycle/rul): {train_df.shape[1] - 3}")


if __name__ == "__main__":
    main()