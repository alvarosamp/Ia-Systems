from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Dict, Tuple
import torch

from src.dl.models_v2 import LSTMRegressor


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT_DIR / "artifacts" / "model" / "lstm.pth"

def _torch_load_weights(path: Path) -> Dict[str, Any]:
    # Prefer the safer weights-only mode when available.
    try:
        return torch.load(path, map_location=DEVICE, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=DEVICE)


def _infer_lstm_arch(state_dict: Dict[str, Any]) -> Tuple[int, int, int]:
    w_ih0 = state_dict.get("lstm.weight_ih_l0")
    w_hh0 = state_dict.get("lstm.weight_hh_l0")
    if w_ih0 is None or w_hh0 is None:
        raise ValueError(
            "State dict inválido: não encontrei chaves 'lstm.weight_ih_l0' e 'lstm.weight_hh_l0'."
        )

    # Shapes:
    # - weight_ih_l0: (4*hidden_size, input_size)
    # - weight_hh_l0: (4*hidden_size, hidden_size)
    input_size = int(w_ih0.shape[1])
    hidden_size = int(w_hh0.shape[1])

    layer_ids = []
    pattern = re.compile(r"^lstm\.weight_ih_l(\d+)$")
    for key in state_dict.keys():
        m = pattern.match(key)
        if m:
            layer_ids.append(int(m.group(1)))
    num_layers = (max(layer_ids) + 1) if layer_ids else 1

    return input_size, hidden_size, num_layers


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

    state_dict = _torch_load_weights(MODEL_PATH)
    if not isinstance(state_dict, dict):
        raise ValueError(f"Unexpected model format in {MODEL_PATH} (expected state_dict dict)")

    feature_dim, hidden_size, num_layers = _infer_lstm_arch(state_dict)

    # Dropout não afeta shape e não existe no state_dict; manter um default razoável.
    dropout = 0.2

    model = LSTMRegressor(
        input_size=feature_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    ).to(DEVICE)

    model.load_state_dict(state_dict)
    model.eval()

    checkpoint = {
        "seq_len": None,  # variável (aceita qualquer comprimento)
        "feature_dim": feature_dim,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "dropout": dropout,
    }

    return model, checkpoint


model, checkpoint = load_model()


def validate_sequence(sequence):
    if not isinstance(sequence, list) or len(sequence) == 0:
        raise ValueError("sequence deve ser uma lista não-vazia")

    feature_dim = checkpoint["feature_dim"]

    for i, row in enumerate(sequence):
        if not isinstance(row, list) or len(row) != feature_dim:
            raise ValueError(f"Row {i}: expected feature_dim={feature_dim}, got {len(row) if isinstance(row, list) else 'non-list'}")


def predict(sequence):
    validate_sequence(sequence)

    x = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = model(x).cpu().numpy().item()

    return float(pred)