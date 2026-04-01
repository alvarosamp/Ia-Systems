from __future__ import annotations

from pathlib import Path
import json

import optuna
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dl.datasets import create_sequences, TurbofanDataset
from src.dl.models import LSTMModel


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT_DIR = Path(__file__).resolve().parents[2]
TRAIN_FILE = "C:\\Users\\vish8\\OneDrive\\Documentos\\GitHub\\Ia-Systems\\turbofan-rul-prediction\\src\\data\\processed\\train_features.parquet"
OPTUNA_DIR = ROOT_DIR / "artifacts" / "optuna"
OPTUNA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OPTUNA_DIR / "lstm_best_params.json"


def objective(trial: optuna.Trial) -> float:
    df = pd.read_parquet(TRAIN_FILE)

    seq_len = trial.suggest_int("seq_len", 20, 60)
    hidden_size = trial.suggest_int("hidden_size", 64, 256)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

    X, y = create_sequences(df, seq_len=seq_len)
    dataset = TurbofanDataset(X, y)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = LSTMModel(input_size=X.shape[2], hidden_size=hidden_size).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()

    final_loss = None

    for _ in range(3):
        epoch_loss = 0.0

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            preds = model(X_batch).squeeze(-1)
            loss = loss_fn(preds, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        final_loss = epoch_loss / len(loader)

    return float(final_loss)


def main():
    print(f"Using device: {DEVICE}")
    print(f"Loading training data from: {TRAIN_FILE}")

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=2000)

    best_result = {
        "best_value": study.best_value,
        "best_params": study.best_params,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(best_result, f, indent=2)

    print("\nBest hyperparameters:")
    print(best_result)
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()