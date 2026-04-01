import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from src.dl.datasets import create_sequences, TurboFanDataset
from src.dl.models import LSTModel, CNNModel, TCNModel, TransformerModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model(name, input_size):
    if name == "LSTM":
        return LSTModel(input_size).to(DEVICE)
    elif name == "CNN":
        return CNNModel(input_size).to(DEVICE)
    elif name == "TCN":
        return TCNModel(input_size).to(DEVICE)
    elif name == "Transformer":
        return TransformerModel(input_size).to(DEVICE)
    else:
        raise ValueError(f"Unknown model name: {name}")

def train(model_name = 'lstm', epoch = 50, seq_len = 30):
    df = pd.read_parquet(
        r"C:\Users\vish8\OneDrive\Documentos\GitHub\Ia-Systems\turbofan-rul-prediction\src\data\processed\test.parquet"
    )
    X, y = create_sequences(df, seq_len)
    dataset = TurboFanDataset(X, y)
    loader = DataLoader(dataset, batch_size = 64, shuffle=True)
    model = get_model(model_name, X.shape[2])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    
    for epoch in range(epoch):
        total_loss = 0
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            preds = model(X_batch).squeeze()
            loss = loss_fn(preds, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epoch}, Loss: {total_loss/len(loader):.4f}")
    torch.save(model.state_dict(), "artifacts/model/lstm.pth")    
if __name__ == "__main__":
    train('LSTM')