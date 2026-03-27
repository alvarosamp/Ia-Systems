from __future__ import annotations
import pandas as pd
from src.core.settings import FEATURE_TRAIN_FILE, FEATURE_TEST_FILE, DROP_COLUMNS

def main():
    train_df = pd.read_parquet(FEATURE_TRAIN_FILE)
    test_df = pd.read_parquet(FEATURE_TEST_FILE)

    train_x = train_df.drop(columns=DROP_COLUMNS, errors="ignore")
    test_x = test_df.drop(columns=DROP_COLUMNS, errors="ignore")

    summary = []

    for col in train_x.columns:
        summary.append(
            {
                "feature": col,
                "train_mean": float(train_x[col].mean()),
                "test_mean": float(test_x[col].mean()),
                "train_std": float(train_x[col].std()),
                "test_std": float(test_x[col].std()),
            }
        )

    result = pd.DataFrame(summary)
    print(result.head(20).to_string(index=False))


if __name__ == "__main__":
    main()