import torch
import numpy as np
from src.dl.models_v2 import LSTMRegressor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(path):
    checkpoint = torch.load(path, map_location=DEVICE)

    model = LSTMRegressor(
        input_size=checkpoint["feature_dim"],
        hidden_size=checkpoint["hidden_size"],
        num_layers=checkpoint["num_layers"],
        dropout=checkpoint["dropout"],
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint

def predict(model, sequence):
    with torch.no_grad():
        x = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        pred = model(x).cpu().numpy().item()
    return pred