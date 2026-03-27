from __future__ import annotations

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

ARTIFACTS_DIR = ROOT_DIR / "artifacts"
MODEL_DIR = ARTIFACTS_DIR / "model"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
OPTUNA_DIR = ARTIFACTS_DIR / "optuna"



# Ajuste para buscar arquivos em src/data/raw/archive/CMaps
SRC_DIR = ROOT_DIR / "src"
ARCHIVE_RAW_DIR = SRC_DIR / "data" / "raw" / "archive" / "CMaps"
TRAIN_RAW_FILE = ARCHIVE_RAW_DIR / "train_FD001.txt"
TEST_RAW_FILE = ARCHIVE_RAW_DIR / "test_FD001.txt"
RUL_RAW_FILE = ARCHIVE_RAW_DIR / "RUL_FD001.txt"

DROP_COLUMNS = ["unit_id", "cycle", "rul"]


# Ajuste para buscar features em src/data/processed/
SRC_PROCESSED_DIR = SRC_DIR / "data" / "processed"
TRAIN_PROCESSED_FILE = SRC_PROCESSED_DIR / "train.parquet"
TEST_PROCESSED_FILE = SRC_PROCESSED_DIR / "test.parquet"
FEATURE_TRAIN_FILE = SRC_PROCESSED_DIR / "train_features.parquet"
FEATURE_TEST_FILE = SRC_PROCESSED_DIR / "test_features.parquet"

BEST_MODEL_FILE = MODEL_DIR / "best_model.joblib"
PREDICTIONS_FILE = MODEL_DIR / "predictions.parquet"
METRICS_FILE = REPORTS_DIR / "metrics.json"
OPTUNA_STUDY_FILE = OPTUNA_DIR / "study_best_params.json"


def ensure_directories() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    OPTUNA_DIR.mkdir(parents=True, exist_ok=True)