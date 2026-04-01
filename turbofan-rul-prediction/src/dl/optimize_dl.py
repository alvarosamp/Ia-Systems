import optuna 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from src.dl.datasets import create_sequences, TurboFanDataset
from src.dl.models import LSTModel

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def objective(trial):
    df = pd.read_parquet(r'C:\Users\vish8\OneDrive\Documentos\GitHub\Ia-Systems\turbofan-rul-prediction\src\data\processed\test.parquet')
    seq_len = trial.suggest_int('seq_len', 20, 60)
    hidden_size = trial.suggest_int('hidden_size', 64, 256)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    X, y = create_sequences(df, seq_len)
    dataset = TurboFanDataset(X, y)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    model = LSTModel(input_size=X.shape[2], hidden_size=hidden_size).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    for _ in range(3):
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

            preds = model(X_batch).squeeze()
            loss = criterion(preds, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return loss.item()

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)
print("Best hyperparameters:", study.best_params)