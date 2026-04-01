from __future__ import annotations

from pathlib import Path
import time

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dl.datasets import create_sequences, TurbofanDataset
from src.dl.models import LSTMModel, CNNModel, TCNModel, TransformerModel


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT_DIR = Path(__file__).resolve().parents[2]
TRAIN_FILE = "C:\\Users\\vish8\\OneDrive\\Documentos\\GitHub\\Ia-Systems\\turbofan-rul-prediction\\src\\data\\processed\\train_features.parquet"
MODEL_DIR = ROOT_DIR / "artifacts" / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def get_model(name: str, input_size: int, hidden_size: int = 128):
    name = name.lower()

    if name == "lstm":
        return LSTMModel(input_size=input_size, hidden_size=hidden_size).to(DEVICE)
    elif name == "cnn":
        return CNNModel(input_size=input_size).to(DEVICE)
    elif name == "tcn":
        return TCNModel(input_size=input_size).to(DEVICE)
    elif name == "transformer":
        return TransformerModel(input_size=input_size).to(DEVICE)
    else:
        raise ValueError(f"Unknown model name: {name}")


def train(model_name: str = "lstm", epochs: int = 10, seq_len: int = 30, hidden_size: int = 128, lr: float = 1e-3):
    print(f"Using device: {DEVICE}")
    print(f"Loading training data from: {TRAIN_FILE}")

    df = pd.read_parquet(TRAIN_FILE)

    X, y = create_sequences(df, seq_len=seq_len)
    dataset = TurbofanDataset(X, y)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = get_model(model_name, input_size=X.shape[2], hidden_size=hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    start_time = time.time()

    for epoch_idx in range(epochs):
        model.train()
        total_loss = 0.0

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            preds = model(X_batch).squeeze(-1)
            loss = loss_fn(preds, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch_idx + 1}/{epochs} - Loss: {avg_loss:.4f}")

    elapsed = time.time() - start_time
    print(f"Training finished in {elapsed:.2f} seconds")

    model_path = MODEL_DIR / f"{model_name.lower()}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    train(model_name="lstm", epochs=5000, seq_len=30, hidden_size=128, lr=1e-3)