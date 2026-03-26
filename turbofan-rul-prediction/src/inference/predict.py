from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd



# Ajuste para refletir a estrutura real do projeto
SRC_DIR = Path(__file__).resolve().parent
MODEL_FILE = SRC_DIR.parent.parent / "artifacts" / "model" / "model.joblib"
TEST_FILE = SRC_DIR.parent.parent / "data" / "processed" / "test.parquet"
OUTPUT_FILE = SRC_DIR.parent.parent / "artifacts" / "model" / "predictions.parquet"

DROP_COLUMNS = ["unit_id", "cycle", "rul"]


def load_model():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {MODEL_FILE}")
    return joblib.load(MODEL_FILE)


def load_data(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Arquivo de entrada não encontrado: {input_path}")
    return pd.read_parquet(input_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=str(TEST_FILE))
    parser.add_argument("--output", type=str, default=str(OUTPUT_FILE))
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = load_model()
    df = load_data(input_path)

    x = df.drop(columns=DROP_COLUMNS, errors="ignore")
    preds = model.predict(x)

    result = df[["unit_id", "cycle", "rul"]].copy()
    result["predicted_rul"] = preds
    result.to_parquet(output_path, index=False)

    print(
        json.dumps(
            {
                "input": str(input_path),
                "output": str(output_path),
                "rows": len(result),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()