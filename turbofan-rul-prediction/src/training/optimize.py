from __future__ import annotations

import json
from pathlib import Path

import hydra
import mlflow
import optuna
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.core.settings import ensure_directories, DROP_COLUMNS




def load_training_data(train_path: str):
    df = pd.read_parquet(train_path)
    x = df.drop(columns=DROP_COLUMNS, errors="ignore")
    y = df["rul"]
    return x, y


def rf_objective(trial, x, y, random_state: int):
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=trial.suggest_int("n_estimators", 100, 500),
                    max_depth=trial.suggest_int("max_depth", 4, 20),
                    min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
                    min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 5),
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    cv = KFold(n_splits=3, shuffle=True, random_state=random_state)
    score = cross_val_score(
        model,
        x,
        y,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=-1,
    ).mean()
    return -score


def xgb_objective(trial, x, y, random_state: int):
    model = XGBRegressor(
        n_estimators=trial.suggest_int("n_estimators", 100, 500),
        max_depth=trial.suggest_int("max_depth", 3, 12),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        random_state=random_state,
        n_jobs=-1,
        objective="reg:squarederror",
    )

    cv = KFold(n_splits=3, shuffle=True, random_state=random_state)
    score = cross_val_score(
        model,
        x,
        y,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=-1,
    ).mean()
    return -score


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    ensure_directories()

    project_root = Path(hydra.utils.get_original_cwd())
    train_path = project_root / cfg.paths.train_data
    optuna_output = project_root / cfg.paths.optuna_output

    print("Config carregada no optimize:")
    print(OmegaConf.to_yaml(cfg))

    x, y = load_training_data(str(train_path))

    mlflow.set_experiment(f"{cfg.mlflow.experiment_name}-optuna")

    results = {}

    with mlflow.start_run(run_name="optuna_rf_xgb"):
        rf_study = optuna.create_study(direction="minimize")
        rf_study.optimize(
            lambda trial: rf_objective(trial, x, y, cfg.training.random_state),
            n_trials=cfg.optuna.n_trials,
        )

        xgb_study = optuna.create_study(direction="minimize")
        xgb_study.optimize(
            lambda trial: xgb_objective(trial, x, y, cfg.training.random_state),
            n_trials=cfg.optuna.n_trials,
        )

        results["random_forest"] = {
            "best_value_rmse": rf_study.best_value,
            "best_params": rf_study.best_params,
        }
        results["xgboost"] = {
            "best_value_rmse": xgb_study.best_value,
            "best_params": xgb_study.best_params,
        }

        optuna_output.parent.mkdir(parents=True, exist_ok=True)
        with open(optuna_output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        mlflow.log_dict(results, "optuna_results.json")

        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()