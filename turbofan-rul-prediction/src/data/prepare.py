from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

from src.core.schemas import validate_raw_processed

from src.core.settings import (
    TRAIN_RAW_FILE,
    TEST_RAW_FILE,
    RUL_RAW_FILE,
    TRAIN_PROCESSED_FILE,
    TEST_PROCESSED_FILE,
    ensure_directories,
)


def build_column_names() -> List[str]:
    operational_settings = [f"setting_{i}" for i in range(1, 4)]
    sensors = [f"sensor_{i}" for i in range(1, 22)]
    return ["unit_id", "cycle", *operational_settings, *sensors]


def read_cmapss_file(file_path: Path) -> pd.DataFrame:
    columns = build_column_names()
    df = pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None,
        names=columns,
        engine="python",
    )
    return df


def compute_train_rul(train_df: pd.DataFrame) -> pd.DataFrame:
    max_cycle = train_df.groupby("unit_id")["cycle"].max().rename("max_cycle")
    train_df = train_df.merge(max_cycle, on="unit_id", how="left")
    train_df["rul"] = train_df["max_cycle"] - train_df["cycle"]
    train_df = train_df.drop(columns=["max_cycle"])
    return train_df


def compute_test_rul(test_df: pd.DataFrame, rul_df: pd.DataFrame) -> pd.DataFrame:
    rul_df = rul_df.copy()
    rul_df.columns = ["extra_rul"]
    rul_df["unit_id"] = range(1, len(rul_df) + 1)

    max_cycle = test_df.groupby("unit_id")["cycle"].max().rename("max_cycle")
    test_df = test_df.merge(max_cycle, on="unit_id", how="left")
    test_df = test_df.merge(rul_df, on="unit_id", how="left")
    test_df["rul"] = test_df["max_cycle"] - test_df["cycle"] + test_df["extra_rul"]
    test_df = test_df.drop(columns=["max_cycle", "extra_rul"])
    return test_df


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop_duplicates()
    df = df.sort_values(["unit_id", "cycle"]).reset_index(drop=True)
    return df


def main() -> None:
    ensure_directories()

    if not TRAIN_RAW_FILE.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {TRAIN_RAW_FILE}")
    if not TEST_RAW_FILE.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {TEST_RAW_FILE}")
    if not RUL_RAW_FILE.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {RUL_RAW_FILE}")

    train_df = read_cmapss_file(TRAIN_RAW_FILE)
    test_df = read_cmapss_file(TEST_RAW_FILE)
    rul_df = pd.read_csv(RUL_RAW_FILE, sep=r"\s+", header=None, engine="python").dropna(axis=1, how="all")

    train_df = clean_dataframe(train_df)
    test_df = clean_dataframe(test_df)

    train_df = compute_train_rul(train_df)
    test_df = compute_test_rul(test_df, rul_df)
    
    train_df = validate_raw_processed(train_df, name="train")
    test_df = validate_raw_processed(test_df, name="test")

    train_df.to_parquet(TRAIN_PROCESSED_FILE, index=False)
    test_df.to_parquet(TEST_PROCESSED_FILE, index=False)

    print(f"Train salvo em: {TRAIN_PROCESSED_FILE}")
    print(f"Test salvo em: {TEST_PROCESSED_FILE}")
    print(f"Shape train: {train_df.shape}")
    print(f"Shape test: {test_df.shape}")


if __name__ == "__main__":
    main()