from pathlib import Path
import torch
import numpy as np

from src.dl.models_v2 import LSTMRegressor


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = Path("C:\\Users\\vish8\\OneDrive\\Documentos\\GitHub\\Ia-Systems\\turbofan-rul-prediction\\src\\artifacts\\model\\lstm_v2.pth")


def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    model = LSTMRegressor(
        input_size=checkpoint["feature_dim"],
        hidden_size=checkpoint["hidden_size"],
        num_layers=checkpoint["num_layers"],
        dropout=checkpoint["dropout"],
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint


model, checkpoint = load_model()


def validate_sequence(sequence):
    seq_len = checkpoint["seq_len"]
    feature_dim = checkpoint["feature_dim"]

    if len(sequence) != seq_len:
        raise ValueError(f"Expected seq_len={seq_len}, got {len(sequence)}")

    if len(sequence[0]) != feature_dim:
        raise ValueError(f"Expected feature_dim={feature_dim}, got {len(sequence[0])}")


def predict(sequence):
    seq_len = checkpoint["seq_len"]
    feature_dim = checkpoint["feature_dim"]

    # Pad or truncate sequence as needed
    if len(sequence) < seq_len:
        # Pad with zeros
        padding = [[0.0] * feature_dim] * (seq_len - len(sequence))
        sequence = padding + sequence  # pad at the beginning
    elif len(sequence) > seq_len:
        # Truncate from the start
        sequence = sequence[-seq_len:]

    # Optionally, still validate feature_dim
    if any(len(row) != feature_dim for row in sequence):
        raise ValueError(f"Expected feature_dim={feature_dim}, got a row with different length")

    x = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = model(x).cpu().numpy().item()

    return float(pred)