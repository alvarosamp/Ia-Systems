from __future__ import annotations

from pathlib import Path

# =========================
# RAIZ DO PROJETO
# =========================
ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"

# =========================
# DADOS
# =========================
# Dados brutos (NASA C-MAPSS) — ficam dentro de src/data por enquanto
RAW_DIR = SRC_DIR / "data" / "raw" / "archive" / "CMaps"
TRAIN_RAW_FILE = RAW_DIR / "train_FD001.txt"
TEST_RAW_FILE = RAW_DIR / "test_FD001.txt"
RUL_RAW_FILE = RAW_DIR / "RUL_FD001.txt"

# Dados processados
PROCESSED_DIR = SRC_DIR / "data" / "processed"
TRAIN_PROCESSED_FILE = PROCESSED_DIR / "train.parquet"
TEST_PROCESSED_FILE = PROCESSED_DIR / "test.parquet"
FEATURE_TRAIN_FILE = PROCESSED_DIR / "train_features.parquet"
FEATURE_TEST_FILE = PROCESSED_DIR / "test_features.parquet"

# =========================
# ARTEFATOS (modelos, métricas, optuna)
# =========================
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
MODEL_DIR = ARTIFACTS_DIR / "model"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
OPTUNA_DIR = ARTIFACTS_DIR / "optuna"

# Sklearn (XGBoost / RF) — baseline
BEST_MODEL_FILE = MODEL_DIR / "best_model.joblib"
PREDICTIONS_FILE = MODEL_DIR / "predictions.parquet"
METRICS_FILE = REPORTS_DIR / "metrics.json"

# LSTM — modelo principal servido pela API
LSTM_MODEL_FILE = MODEL_DIR / "lstm.pth"
LSTM_METRICS_FILE = REPORTS_DIR / "lstm_metrics.json"
LSTM_HISTORY_FILE = REPORTS_DIR / "lstm_history.json"

# Optuna
OPTUNA_SKLEARN_FILE = OPTUNA_DIR / "sklearn_best_params.json"
OPTUNA_LSTM_FILE = OPTUNA_DIR / "lstm_best_params.json"

# =========================
# COLUNAS
# =========================
DROP_COLUMNS = ["unit_id", "cycle", "rul"]


def ensure_directories() -> None:
    """Garante que todos os diretórios de saída existam."""
    for d in (PROCESSED_DIR, MODEL_DIR, REPORTS_DIR, OPTUNA_DIR):
        d.mkdir(parents=True, exist_ok=True)