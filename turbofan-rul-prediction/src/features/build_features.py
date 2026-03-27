from __future__ import annotations

import pandas as pd

from src.core.settings import (
    TRAIN_PROCESSED_FILE,
    TEST_PROCESSED_FILE,
    FEATURE_TRAIN_FILE,
    FEATURE_TEST_FILE,
    ensure_directories,
)


SENSOR_COLUMNS = [f"sensor_{i}" for i in range(1, 22)]
SETTING_COLUMNS = [f"setting_{i}" for i in range(1, 4)]

def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in SENSOR_COLUMNS:
        df[f"{col}_rolling_mean_5"] = (
            df.groupby("unit_id")[col].rolling(window=5, min_periods=1).mean().reset_index(level=0, drop=True)
        )
        df[f"{col}_rolling_std_5"] = (
            df.groupby("unit_id")[col].rolling(window=5, min_periods=1).std().reset_index(level=0, drop=True).fillna(0)
        )

    df["cycle_ratio"] = df["cycle"] / df.groupby("unit_id")["cycle"].transform("max")
    return df

def remove_constant_columns(df: pd.DataFrame) -> pd.DataFrame:
    nunique = df.nunique()
    constant_cols = nunique[nunique <=1].index.tolist()
    return df.drop(columns=constant_cols)

def main() -> None:
    ensure_directories()
    train_df = pd.read_parquet(TRAIN_PROCESSED_FILE)
    test_df = pd.read_parquet(TEST_PROCESSED_FILE)
    train_df = add_basic_features(train_df)
    test_df = add_basic_features(test_df)
    combined = pd.concat([train_df.drop(columns = ['rul']),test_df.drop(columns = ['rul'])], axis=0)
    combined = remove_constant_columns(combined)
    common_cols = combined.columns.to_list()
    train_df = train_df[[c for c in common_cols if c in train_df.columns] + ["rul"]]
    test_df = test_df[[c for c in common_cols if c in test_df.columns] + ["rul"]]
    train_df.to_parquet(FEATURE_TRAIN_FILE, index=False)
    test_df.to_parquet(FEATURE_TEST_FILE, index=False)
    
    print(f"Features saved to {FEATURE_TRAIN_FILE} and {FEATURE_TEST_FILE}")
    
if __name__ == "__main__":
    main()