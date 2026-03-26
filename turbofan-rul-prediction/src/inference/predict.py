from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd


# =========================
# PATHS DO PROJETO
# =========================
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent.parent

PROCESSED_DIR = SRC_DIR.parent / "data" / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "model"

MODEL_FILE = ARTIFACTS_DIR / "model.joblib"
TEST_FILE = PROCESSED_DIR / "test.parquet"
OUTPUT_FILE = ARTIFACTS_DIR / "predictions.parquet"

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
    parser = argparse.ArgumentParser(description="Gera predições de RUL usando o modelo treinado.")
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

    result = df.copy()
    result["predicted_rul"] = preds

    # Se essas colunas existirem, organiza melhor a saída
    preferred_cols = ["unit_id", "cycle", "rul", "predicted_rul"]
    existing_preferred = [col for col in preferred_cols if col in result.columns]
    remaining_cols = [col for col in result.columns if col not in existing_preferred]
    result = result[existing_preferred + remaining_cols]

    result.to_parquet(output_path, index=False)

    print(
        json.dumps(
            {
                "model": str(MODEL_FILE),
                "input": str(input_path),
                "output": str(output_path),
                "rows": len(result),
                "columns_used_for_prediction": list(x.columns),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()