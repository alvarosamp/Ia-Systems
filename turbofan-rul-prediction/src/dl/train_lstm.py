from __future__ import annotations

import json
import time

import hydra
import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader

from src.core.settings import (
    FEATURE_TRAIN_FILE,
    FEATURE_TEST_FILE,
    LSTM_MODEL_FILE,
    LSTM_METRICS_FILE,
    LSTM_HISTORY_FILE,
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


def compute_metrics(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def evaluate_model(model, loader, loss_fn):
    model.eval()
    losses, preds_all, targets_all = [], [], []

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


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    if cfg.model.name != "lstm":
        raise ValueError(f"train_lstm.py espera model=lstm, recebido model={cfg.model.name}")

    ensure_directories()

    print("Config carregada:")
    print(OmegaConf.to_yaml(cfg))

    p = cfg.model.params
    t = cfg.model.training

    # =========================
    # DADOS
    # =========================
    train_full_df = pd.read_parquet(FEATURE_TRAIN_FILE)
    test_df = pd.read_parquet(FEATURE_TEST_FILE)

    train_df, val_df = split_train_val_by_unit(train_full_df, val_ratio=t.val_ratio)

    scaler, feature_cols = fit_feature_scaler(train_df)
    train_df = apply_feature_scaler(train_df, scaler, feature_cols)
    val_df = apply_feature_scaler(val_df, scaler, feature_cols)
    test_df = apply_feature_scaler(test_df, scaler, feature_cols)

    X_train, y_train = create_sequences(train_df, seq_len=p.seq_len)
    X_val, y_val = create_sequences(val_df, seq_len=p.seq_len)
    X_test, y_test = create_sequences(test_df, seq_len=p.seq_len)

    train_loader = DataLoader(TurbofanDataset(X_train, y_train), batch_size=p.batch_size, shuffle=True)
    val_loader = DataLoader(TurbofanDataset(X_val, y_val), batch_size=p.batch_size, shuffle=False)
    test_loader = DataLoader(TurbofanDataset(X_test, y_test), batch_size=p.batch_size, shuffle=False)

    # =========================
    # MODELO
    # =========================
    model = LSTMRegressor(
        input_size=X_train.shape[2],
        hidden_size=p.hidden_size,
        num_layers=p.num_layers,
        dropout=p.dropout,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=p.lr, weight_decay=p.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=t.scheduler_factor, patience=t.scheduler_patience
    )
    loss_fn = nn.MSELoss()

    # =========================
    # MLFLOW
    # =========================
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    with mlflow.start_run(run_name="lstm") as run:
        # Loga params
        mlflow.log_param("model_name", "lstm")
        mlflow.log_param("device", str(DEVICE))
        mlflow.log_param("feature_dim", X_train.shape[2])
        mlflow.log_param("n_train_sequences", len(X_train))
        mlflow.log_param("n_val_sequences", len(X_val))
        mlflow.log_param("n_test_sequences", len(X_test))
        for k, v in dict(p).items():
            mlflow.log_param(k, v)
        for k, v in dict(t).items():
            mlflow.log_param(f"train_{k}", v)

        # =========================
        # TREINO
        # =========================
        best_val_rmse = float("inf")
        best_state = None
        wait = 0
        history = []
        start_time = time.time()

        for epoch in range(1, t.epochs + 1):
            model.train()
            train_losses = []

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)

                preds = model(X_batch)
                loss = loss_fn(preds, y_batch)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=t.grad_clip)
                optimizer.step()

                train_losses.append(loss.item())

            train_loss = float(np.mean(train_losses))
            val_metrics = evaluate_model(model, val_loader, loss_fn)
            scheduler.step(val_metrics["rmse"])

            current_lr = optimizer.param_groups[0]["lr"]

            history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_rmse": val_metrics["rmse"],
                "val_mae": val_metrics["mae"],
                "val_r2": val_metrics["r2"],
                "lr": current_lr,
            })

            # Loga métricas por época no MLflow
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_metrics["loss"], step=epoch)
            mlflow.log_metric("val_rmse", val_metrics["rmse"], step=epoch)
            mlflow.log_metric("val_mae", val_metrics["mae"], step=epoch)
            mlflow.log_metric("val_r2", val_metrics["r2"], step=epoch)
            mlflow.log_metric("lr", current_lr, step=epoch)

            print(
                f"Epoch {epoch}/{t.epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"val_rmse={val_metrics['rmse']:.4f} | "
                f"val_mae={val_metrics['mae']:.4f} | "
                f"lr={current_lr:.6f}"
            )

            if val_metrics["rmse"] < best_val_rmse:
                best_val_rmse = val_metrics["rmse"]
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1

            if wait >= t.patience:
                print(f"Early stopping at epoch {epoch}")
                break

        elapsed = time.time() - start_time

        # =========================
        # AVALIAÇÃO FINAL
        # =========================
        if best_state is None:
            raise RuntimeError("Nenhum best_state foi salvo durante o treino.")

        model.load_state_dict(best_state)
        test_metrics = evaluate_model(model, test_loader, loss_fn)

        # Loga métricas finais
        mlflow.log_metric("best_val_rmse", best_val_rmse)
        mlflow.log_metric("test_rmse", test_metrics["rmse"])
        mlflow.log_metric("test_mae", test_metrics["mae"])
        mlflow.log_metric("test_r2", test_metrics["r2"])
        mlflow.log_metric("train_time_seconds", elapsed)

        # =========================
        # SALVA CHECKPOINT (com scaler — essencial pra API)
        # =========================
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "seq_len": p.seq_len,
            "hidden_size": p.hidden_size,
            "num_layers": p.num_layers,
            "dropout": p.dropout,
            "feature_dim": X_train.shape[2],
            "feature_cols": feature_cols,
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
        }
        torch.save(checkpoint, LSTM_MODEL_FILE)

        # Salva métricas e history em JSON
        final_report = {
            "best_val_rmse": best_val_rmse,
            "test_metrics": test_metrics,
            "train_time_seconds": elapsed,
            "params": dict(p),
            "training": dict(t),
        }

        with open(LSTM_METRICS_FILE, "w", encoding="utf-8") as f:
            json.dump(final_report, f, indent=2)

        with open(LSTM_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        # Loga artefatos no MLflow
        mlflow.log_artifact(str(LSTM_MODEL_FILE), artifact_path="model")
        mlflow.log_artifact(str(LSTM_METRICS_FILE), artifact_path="reports")
        mlflow.log_artifact(str(LSTM_HISTORY_FILE), artifact_path="reports")

        print("\n=== Final Test Metrics ===")
        print(json.dumps(test_metrics, indent=2))
        print(f"Run ID: {run.info.run_id}")
        print(f"Modelo salvo em: {LSTM_MODEL_FILE}")


if __name__ == "__main__":
    main()