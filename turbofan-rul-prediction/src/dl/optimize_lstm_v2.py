from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader

from src.dl.datasets import (
    TurbofanDataset,
    apply_feature_scaler,
    create_sequences,
    fit_feature_scaler,
    split_train_val_by_unit,
)
from src.dl.models_v2 import LSTMRegressor


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT_DIR = Path(__file__).resolve().parents[2]
TRAIN_FEATURES_FILE = r"C:\Users\vish8\OneDrive\Documentos\GitHub\Ia-Systems\turbofan-rul-prediction\src\data\processed\train_features.parquet"
OUTPUT_FILE = Path(r"C:\Users\vish8\OneDrive\Documentos\GitHub\Ia-Systems\turbofan-rul-prediction\src\artifacts\optuna\lstm_v2_best_params.json")
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)


def evaluate_rmse(model, loader):
    model.eval()
    preds_all = []
    targets_all = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            preds = model(X_batch)
            preds_all.extend(preds.cpu().numpy())
            targets_all.extend(y_batch.cpu().numpy())

    rmse = np.sqrt(mean_squared_error(np.array(targets_all), np.array(preds_all)))
    return float(rmse)


def objective(trial: optuna.Trial):
    df = pd.read_parquet(TRAIN_FEATURES_FILE)
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

    train_ds = TurbofanDataset(X_train, y_train)
    val_ds = TurbofanDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = LSTMRegressor(
        input_size=X_train.shape[2],
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    best_val_rmse = float("inf")

    for _ in range(20):
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

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse

    return best_val_rmse


def main():
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)

    result = {
        "best_value_rmse": study.best_value,
        "best_params": study.best_params,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("\nBest LSTM hyperparameters:")
    print(json.dumps(result, indent=2))
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()