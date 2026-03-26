from __future__ import annotations

import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler



# Ajuste para refletir a estrutura real do projeto
SRC_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = SRC_DIR.parent / "data" / "processed"
ARTIFACTS_DIR = SRC_DIR.parent.parent / "artifacts" / "model"

TRAIN_FILE = PROCESSED_DIR / "train.parquet"
TEST_FILE = PROCESSED_DIR / "test.parquet"

MODEL_FILE = ARTIFACTS_DIR / "model.joblib"
METRICS_FILE = ARTIFACTS_DIR / "metrics.json"

DROP_COLUMNS = ["unit_id", "cycle", "rul"]


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {TRAIN_FILE}")
    if not TEST_FILE.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {TEST_FILE}")

    train_df = pd.read_parquet(TRAIN_FILE)
    test_df = pd.read_parquet(TEST_FILE)
    return train_df, test_df


def build_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    x = df.drop(columns=DROP_COLUMNS, errors="ignore")
    y = df["rul"]
    return x, y


def build_model() -> Pipeline:
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=300,
                    max_depth=12,
                    min_samples_split=4,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    return model


def evaluate(y_true: pd.Series, y_pred) -> dict:
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
    }


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    train_df, test_df = load_data()
    x_train, y_train = build_xy(train_df)
    x_test, y_test = build_xy(test_df)

    model = build_model()

    mlflow.set_experiment("turbofan-rul-prediction")

    with mlflow.start_run(run_name="random_forest_baseline"):
        mlflow.log_param("model_type", "RandomForestRegressor")
        mlflow.log_param("train_rows", len(train_df))
        mlflow.log_param("test_rows", len(test_df))
        mlflow.log_param("n_features", x_train.shape[1])

        model.fit(x_train, y_train)
        preds = model.predict(x_test)

        metrics = evaluate(y_test, preds)
        mlflow.log_metrics(metrics)

        joblib.dump(model, MODEL_FILE)

        with open(METRICS_FILE, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        mlflow.log_artifact(str(MODEL_FILE))
        mlflow.log_artifact(str(METRICS_FILE))
        mlflow.sklearn.log_model(model, artifact_path="sklearn-model")

        print("Treino concluído com sucesso.")
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()