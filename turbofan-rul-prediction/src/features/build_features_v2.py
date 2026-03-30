from __future__ import annotations
import pandas as pd
from src.core.settings import TRAIN_PROCESSED_FILE, TEST_PROCESSED_FILE, FEATURE_TRAIN_FILE, FEATURE_TEST_FILE, ensure_directories

SENSOR_COLUMNS = [f'sensor_{i}' for i in range(1, 22)]

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in SENSOR_COLUMNS:
        grouped = df.groupby("unit_id")[col]

        df[f"{col}_rolling_mean_5"] = (
            grouped.rolling(window=5, min_periods=1).mean().reset_index(level=0, drop=True)
        )
        df[f"{col}_rolling_std_5"] = (
            grouped.rolling(window=5, min_periods=1).std().reset_index(level=0, drop=True).fillna(0)
        )
        df[f"{col}_rolling_min_5"] = (
            grouped.rolling(window=5, min_periods=1).min().reset_index(level=0, drop=True)
        )
        df[f"{col}_rolling_max_5"] = (
            grouped.rolling(window=5, min_periods=1).max().reset_index(level=0, drop=True)
        )

        df[f"{col}_diff_1"] = grouped.diff().fillna(0)
        df[f"{col}_diff_3"] = grouped.diff(3).fillna(0)

    return df

def add_unit_agreggate_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in SENSOR_COLUMNS:
        df[f"{col}_unit_mean"] = df.groupby("unit_id")[col].transform("mean")
        df[f"{col}_unit_std"] = df.groupby("unit_id")[col].transform("std").fillna(0)
        df[f"{col}_unit_min"] = df.groupby("unit_id")[col].transform("min")
        df[f"{col}_unit_max"] = df.groupby("unit_id")[col].transform("max")

    return df

def add_cycle_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    max_cycle_per_unit = df.groupby("unit_id")["cycle"].transform("max")
    df["cycle_ratio"] = df["cycle"] / max_cycle_per_unit
    df["cycle_left_proxy"] = max_cycle_per_unit - df["cycle"]
    df["cycle_squared"] = df["cycle"] ** 2

    return df

def remove_constant_columns(df: pd.DataFrame) -> pd.DataFrame:
    nunique = df.nunique(dropna=False)
    constant_cols = nunique[nunique <= 1].index.tolist()
    return df.drop(columns=constant_cols)

def align_columns(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
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

def main() :
    ensure_directories()
    train_df = pd.read_csv(TRAIN_PROCESSED_FILE)
    test_df = pd.read_csv(TEST_PROCESSED_FILE)
    train_df = add_temporal_features(train_df)
    test_df = add_temporal_features(test_df)
    train_df = add_unit_agreggate_features(train_df)
    test_df = add_unit_agreggate_features(test_df)
    train_df = add_cycle_features(train_df)
    test_df = add_cycle_features(test_df)
    train_df, test_df = align_columns(train_df, test_df)
    train_df.to_parquet(FEATURE_TRAIN_FILE, index=False)
    test_df.to_parquet(FEATURE_TEST_FILE, index=False)

if __name__ == "__main__":
    main()