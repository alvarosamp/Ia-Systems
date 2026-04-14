from __future__ import annotations

import json

import hydra
import mlflow
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader

from src.core.settings import (
    FEATURE_TRAIN_FILE,
    OPTUNA_LSTM_FILE,
    ensure_directories,
)
from src.dl.datasets import (
    TurbofanDataset,
    apply_feature_scaler,
    create_sequences,
    fit_feature_scaler,
    split_train_val_by_unit,
)
from src.dl.models import LSTMRegressor


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS_PER_TRIAL = 60


def evaluate_rmse(model, loader):
    model.eval()
    preds_all, targets_all = [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            preds = model(X_batch)
            preds_all.extend(preds.cpu().numpy())
            targets_all.extend(y_batch.cpu().numpy())

    return float(np.sqrt(mean_squared_error(np.array(targets_all), np.array(preds_all))))


def make_objective(cfg: DictConfig):
    df = pd.read_parquet(FEATURE_TRAIN_FILE)

    def objective(trial: optuna.Trial):
        train_df, val_df = split_train_val_by_unit(df, val_ratio=0.2)
        scaler, feature_cols = fit_feature_scaler(train_df)
        train_df = apply_feature_scaler(train_df, scaler, feature_cols)
        val_df = apply_feature_scaler(val_df, scaler, feature_cols)

        seq_len = trial.suggest_int("seq_len", 20, 60)
        hidden_size = trial.suggest_int("hidden_size", 64, 256)
        num_layers = trial.suggest_int("num_layers", 1, 3)
        dropout = trial.suggest_float("dropout", 0.0, 0.4)
        lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

        X_train, y_train = create_sequences(train_df, seq_len=seq_len)
        X_val, y_val = create_sequences(val_df, seq_len=seq_len)

        train_loader = DataLoader(TurbofanDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TurbofanDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

        model = LSTMRegressor(
            input_size=X_train.shape[2],
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        ).to(DEVICE)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.MSELoss()

        best_val_rmse = float("inf")

        for _ in range(EPOCHS_PER_TRIAL):
            model.train()
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)
                preds = model(X_batch)
                loss = loss_fn(preds, y_batch)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            val_rmse = evaluate_rmse(model, val_loader)
            best_val_rmse = min(best_val_rmse, val_rmse)

            trial.report(val_rmse, step=_)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return best_val_rmse

    return objective


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    ensure_directories()

    print("Config Optuna LSTM:")
    print(OmegaConf.to_yaml(cfg))

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(f"{cfg.mlflow.experiment_name}-optuna")

    with mlflow.start_run(run_name="optuna_lstm"):
        study = optuna.create_study(
            direction="minimize",
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
        )
        study.optimize(make_objective(cfg), n_trials=cfg.optuna.n_trials)

        result = {
            "best_value_rmse": study.best_value,
            "best_params": study.best_params,
            "n_trials": cfg.optuna.n_trials,
        }

        with open(OPTUNA_LSTM_FILE, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        mlflow.log_dict(result, "lstm_optuna_results.json")
        mlflow.log_metric("best_val_rmse", study.best_value)
        for k, v in study.best_params.items():
            mlflow.log_param(f"best_{k}", v)

        print("\n=== Best LSTM Hyperparameters ===")
        print(json.dumps(result, indent=2))
        print(f"Salvo em: {OPTUNA_LSTM_FILE}")


if __name__ == "__main__":
    main()