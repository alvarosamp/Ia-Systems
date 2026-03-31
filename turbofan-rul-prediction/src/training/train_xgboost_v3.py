from __future__ import annotations
import json
from pathlib import Path
import hydra
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from src.core.settings import ensure_directories, DROP_COLUMNS
from src.training.evaluate import evaluate_regression


def load_data(train_path: str, test_path: str):
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    return train_df, test_df 

def split_xy(df: pd.DataFrame):
    X = df.drop(columns=DROP_COLUMNS, errors = 'ignore')
    y = df["rul"]
    return X, y

def build_model(cfg: DictConfig):
    if cfg.model_name == 'random_forest':
        return Pipeline(
            steps=[
                ('scaler', StandardScaler()),
                ('model', RandomForestRegressor(
                    **cfg.model.params,
                    random_state = cfg.training.random_state,
                    n_jobs = -1,
                ),
            ),
        ]
    )
    if cfg.model_name == 'xgboost':
        return XGBRegressor(
            **cfg.model.params,
            random_state = cfg.training.random_state,
            n_jobs = -1,
            objective = 'reg:squarederror',
        )
    raise ValueError(f"Unsupported model name: {cfg.model_name}")

@hydra.main(version_base = None, config_path = "../../configs", config_name = "train_config")
def main(cfg: DictConfig):
    ensure_directories()

    project_root = Path(hydra.utils.get_original_cwd())

    train_path = project_root / cfg.paths.train_data
    test_path = project_root / cfg.paths.test_data
    model_output_dir = project_root / cfg.paths.model_output

    print("Config carregada:")
    print(OmegaConf.to_yaml(cfg))

    train_df, test_df = load_data(str(train_path), str(test_path))
    x_train, y_train = split_xy(train_df)
    x_test, y_test = split_xy(test_df)

    model = build_model(cfg)

    mlflow.set_experiment(cfg.mlflow.experiment_name)

    with mlflow.start_run(run_name=cfg.model.name):
        mlflow.log_param("model_name", cfg.model.name)
        mlflow.log_params(dict(cfg.model.params))

        model.fit(x_train, y_train)
        preds = model.predict(x_test)

        metrics = evaluate_regression(y_test, preds)
        mlflow.log_metrics(metrics)

        model_output_dir.mkdir(parents=True, exist_ok=True)

        model_file = model_output_dir / f"{cfg.model.name}.joblib"
        metrics_file = model_output_dir / f"metrics_{cfg.model.name}.json"

        print(f"[DEBUG] Salvando métricas em: {metrics_file}")

        joblib.dump(model, model_file)

        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        mlflow.log_artifact(str(metrics_file))
        mlflow.sklearn.log_model(model, artifact_path="model")

        print("Treino concluído.")
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()