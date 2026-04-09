from __future__ import annotations

import argparse
import json

import joblib
import pandas as pd

from src.core.settings import DROP_COLUMNS, BEST_MODEL_FILE, FEATURE_TEST_FILE, PREDICTIONS_FILE, ensure_directories





def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=str(FEATURE_TEST_FILE))
    parser.add_argument("--output", type=str, default=str(PREDICTIONS_FILE))
    args = parser.parse_args()

    ensure_directories()

    model = joblib.load(BEST_MODEL_FILE)
    df = pd.read_parquet(args.input)

    x = df.drop(columns=DROP_COLUMNS, errors="ignore")
    preds = model.predict(x)

    result = df[["unit_id", "cycle", "rul"]].copy()
    result["predicted_rul"] = preds
    result.to_parquet(args.output, index=False)

    print(json.dumps({"rows": len(result), "output": args.output}, indent=2))


if __name__ == "__main__":
    main()