from __feature__ import annotations
import json
from pathlib import Path
import hydra
import mlflow
import optuna
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from sklearn.model.selection import KFold, cross_val_score
from xgboost import XGBRegressor
from src.core.settings import ensure_directories, DROP_COLUMNS

def load_training_data(train_path: str):
    df = pd.read_parquet(train_path)
    X = df.drop(columns=DROP_COLUMNS, errors = 'ignore')
    y = df["rul"]
    return X, y

def xgb_objective(trial, X, y, random_state: int):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "random_state": random_state,
    }
    model = XGBRegressor(**params)
    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
    scores = cross_val_score(model, X, y, cv=kf, scoring="neg_mean_squared_error")
    return -scores.mean()

@hydra.main(version_base = None, config_path = "../../configs", config_name = "optimize_config")
def main(cfg: DictConfig):
    ensure_directories()
    project_tool = Path(hydra.utils.get_original_cwd())
    train_path = project_tool / cfg.paths.train_data
    optuna_output = project_root / cfg.paths.optuna_output
    print(OmegaConf.to_yaml(cfg))

    X, y = load_training_data(train_path)
    mlflow.set_experiment(f'{cfg.mlflow.experiment_name}_optuna')
    with mlflow.start_run(run_name = 'xgboost_optuna'):
        study = optuna.create_study(direction = 'minimize')
        study.optimize(
            lambda trial: xgb_objective(trial, X, y, cfg.random_state),
            n_trials = cfg.optuna.n_trials,
            show_progress_bar = True
        )
        results = {
            'best_value_rmse': study.best_value,
            'best_params': study.best_params
        }
        optuna_output.parent.mkdir(parents = True, exist_ok = True)
        with open(optuna_output, 'w', encoding = 'utf-8') as f:
            json.dump(results, f, indent= 2)
        mlflow.log_dict(results, 'xgboost_optuna_results.json')
        print(json.dumps(results, indent = 2))

    if __name__ == "__main__":
        main()