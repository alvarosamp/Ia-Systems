from __future__ import annotations

import json
from pathlib import Path
from pyexpat import model
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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
TEST_FEATURES_FILE = r"C:\Users\vish8\OneDrive\Documentos\GitHub\Ia-Systems\turbofan-rul-prediction\src\data\processed\test_features.parquet"

MODEL_DIR = Path(r"C:\Users\vish8\OneDrive\Documentos\GitHub\Ia-Systems\turbofan-rul-prediction\src\artifacts\model")
REPORT_DIR = Path(r"C:\Users\vish8\OneDrive\Documentos\GitHub\Ia-Systems\turbofan-rul-prediction\src\artifacts\reports")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def compute_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"rmse": float(rmse), "mae": float(mae), "r2": float(r2)}


def evaluate_model(model, loader, loss_fn):
    model.eval()
    losses = []
    preds_all = []
    targets_all = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            preds = model(X_batch)
            loss = loss_fn(preds, y_batch)

            losses.append(loss.item())
            preds_all.extend(preds.cpu().numpy())
            targets_all.extend(y_batch.cpu().numpy())

    metrics = compute_metrics(np.array(targets_all), np.array(preds_all))
    metrics["loss"] = float(np.mean(losses))
    return metrics


def train_lstm(
    seq_len=53,
    hidden_size=200,
    num_layers=3,
    dropout=0.0068359429004963175,
    lr=0.000780088047989325,
    batch_size=64,
    epochs=300,
    patience=10,
    weight_decay=1.7231272399285946e-05,
):
    print(f"Using device: {DEVICE}")

    train_full_df = pd.read_parquet(TRAIN_FEATURES_FILE)
    test_df = pd.read_parquet(TEST_FEATURES_FILE)

    train_df, val_df = split_train_val_by_unit(train_full_df, val_ratio=0.2)

    scaler, feature_cols = fit_feature_scaler(train_df)
    train_df = apply_feature_scaler(train_df, scaler, feature_cols)
    val_df = apply_feature_scaler(val_df, scaler, feature_cols)
    test_df = apply_feature_scaler(test_df, scaler, feature_cols)

    X_train, y_train = create_sequences(train_df, seq_len=seq_len)
    X_val, y_val = create_sequences(val_df, seq_len=seq_len)
    X_test, y_test = create_sequences(test_df, seq_len=seq_len)

    train_ds = TurbofanDataset(X_train, y_train)
    val_ds = TurbofanDataset(X_val, y_val)
    test_ds = TurbofanDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = LSTMRegressor(
        input_size=X_train.shape[2],
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4
    )
    loss_fn = nn.MSELoss()

    best_val_rmse = float("inf")
    best_state = None
    wait = 0

    history = []
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            preds = model(X_batch)
            loss = loss_fn(preds, y_batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())

        train_loss = float(np.mean(train_losses))
        val_metrics = evaluate_model(model, val_loader, loss_fn)
        scheduler.step(val_metrics["rmse"])

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_rmse": val_metrics["rmse"],
                "val_mae": val_metrics["mae"],
                "val_r2": val_metrics["r2"],
                "lr": optimizer.param_groups[0]["lr"],
            }
        )

        print(
            f"Epoch {epoch}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_rmse={val_metrics['rmse']:.4f} | "
            f"val_mae={val_metrics['mae']:.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.6f}"
        )

        if val_metrics["rmse"] < best_val_rmse:
            best_val_rmse = val_metrics["rmse"]
            best_state = model.state_dict()
            wait = 0
        else:
            wait += 1

        if wait >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    elapsed = time.time() - start_time

    if best_state is None:
        raise RuntimeError("No best model state was saved.")

    model.load_state_dict(best_state)

    test_metrics = evaluate_model(model, test_loader, loss_fn)

    model_path = MODEL_DIR / "lstm_v2.pth"
    metrics_path = REPORT_DIR / "lstm_v2_metrics.json"
    history_path = REPORT_DIR / "lstm_v2_history.json"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "seq_len": seq_len,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "feature_dim": X_train.shape[2],
            "feature_cols": feature_cols,
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
        },
        model_path,
    )

    final_report = {
        "best_val_rmse": best_val_rmse,
        "test_metrics": test_metrics,
        "train_time_seconds": elapsed,
        "params": {
            "seq_len": seq_len,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "lr": lr,
            "batch_size": batch_size,
            "epochs": epochs,
            "patience": patience,
            "weight_decay": weight_decay,
        },
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)

    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print("\nFinal test metrics:")
    print(json.dumps(test_metrics, indent=2))
    print(f"Model saved to: {model_path}")
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    train_lstm(
    seq_len=53,
    hidden_size=200,
    num_layers=3,
    dropout=0.0068359429004963175,
    lr=0.000780088047989325,
    batch_size=64,
    epochs=1000,
    patience=200,
    weight_decay=1.7231272399285946e-05,
)