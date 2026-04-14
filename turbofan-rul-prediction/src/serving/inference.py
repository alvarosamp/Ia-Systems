from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping, Tuple

import numpy as np
import torch
import pandas as pd

from src.core.settings import DROP_COLUMNS, FEATURE_TRAIN_FILE, LSTM_MODEL_FILE, REPORTS_DIR
from src.dl.models import LSTMRegressor


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_checkpoint(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Modelo LSTM não encontrado em: {path}")
    # weights_only=False porque o checkpoint contém metadados (scaler, feature_cols, etc)
    return torch.load(path, map_location=DEVICE, weights_only=False)


def _infer_lstm_params_from_state_dict(state_dict: Mapping[str, Any]) -> Tuple[int, int, int]:
    """Infere (feature_dim, hidden_size, num_layers) a partir de um state_dict salvo."""
    w0 = state_dict.get("lstm.weight_ih_l0")
    if w0 is None or not hasattr(w0, "shape") or len(w0.shape) != 2:
        raise ValueError(
            "Checkpoint/estado do modelo não contém 'lstm.weight_ih_l0' com shape 2D; "
            "não foi possível inferir a arquitetura."
        )

    # Em PyTorch LSTM: weight_ih_l{k} tem shape (4*hidden_size, input_size)
    hidden_size = int(w0.shape[0] // 4)
    feature_dim = int(w0.shape[1])

    layer_indices: set[int] = set()
    for key in state_dict.keys():
        m = re.fullmatch(r"lstm\.weight_ih_l(\d+)", str(key))
        if m:
            layer_indices.add(int(m.group(1)))

    num_layers = (max(layer_indices) + 1) if layer_indices else 1
    return feature_dim, hidden_size, num_layers


def _load_seq_len_hint() -> int | None:
    """Tenta recuperar um seq_len a partir de um payload de exemplo (se existir)."""
    sample_path = REPORTS_DIR / "sample_api_payload.json"
    if not sample_path.exists():
        return None
    try:
        obj = json.loads(sample_path.read_text(encoding="utf-8"))
        seq = obj.get("sequence")
        if isinstance(seq, list) and len(seq) > 0:
            return int(len(seq))
    except Exception:
        return None
    return None


def _derive_feature_cols_and_scaler_from_train(feature_dim: int) -> Tuple[list[str], np.ndarray, np.ndarray]:
    if not FEATURE_TRAIN_FILE.exists():
        raise FileNotFoundError(
            "Faltam metadados do scaler no checkpoint e o arquivo de treino processado não foi encontrado em: "
            f"{FEATURE_TRAIN_FILE}"
        )

    df = pd.read_parquet(FEATURE_TRAIN_FILE)
    feature_cols = [c for c in df.columns if c not in DROP_COLUMNS]

    if len(feature_cols) != feature_dim:
        raise ValueError(
            "Inconsistência entre o modelo e os dados: "
            f"feature_dim inferido do state_dict={feature_dim}, mas o dataset tem {len(feature_cols)} colunas de features. "
            f"(DROP_COLUMNS={DROP_COLUMNS})"
        )

    X = df[feature_cols].to_numpy(dtype=np.float32, copy=False)
    scaler_mean = X.mean(axis=0)
    scaler_scale = X.std(axis=0, ddof=0)

    # Evita divisão por zero caso alguma feature tenha variância zero
    scaler_scale = np.where(scaler_scale == 0.0, 1.0, scaler_scale)
    return feature_cols, scaler_mean.astype(np.float32), scaler_scale.astype(np.float32)


def load_model():
    """
    Carrega o LSTM treinado + metadados do checkpoint.
    O checkpoint deve conter: model_state_dict, seq_len, hidden_size, num_layers,
    dropout, feature_dim, feature_cols, scaler_mean, scaler_scale.
    """
    ckpt = _load_checkpoint(LSTM_MODEL_FILE)

    # Caso 1: checkpoint completo (dict com metadados)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        required_keys = {
            "model_state_dict",
            "seq_len",
            "hidden_size",
            "num_layers",
            "dropout",
            "feature_dim",
            "feature_cols",
            "scaler_mean",
            "scaler_scale",
        }
        missing = required_keys - set(ckpt.keys())
        if missing:
            raise ValueError(f"Checkpoint inválido. Chaves faltando: {missing}")

        model = LSTMRegressor(
            input_size=ckpt["feature_dim"],
            hidden_size=ckpt["hidden_size"],
            num_layers=ckpt["num_layers"],
            dropout=ckpt["dropout"],
        ).to(DEVICE)

        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        metadata = {
            "seq_len": ckpt["seq_len"],
            "feature_dim": ckpt["feature_dim"],
            "hidden_size": ckpt["hidden_size"],
            "num_layers": ckpt["num_layers"],
            "dropout": ckpt["dropout"],
            "feature_cols": ckpt["feature_cols"],
            "scaler_mean": np.array(ckpt["scaler_mean"], dtype=np.float32),
            "scaler_scale": np.array(ckpt["scaler_scale"], dtype=np.float32),
        }

        return model, metadata

    # Caso 2: state_dict puro (OrderedDict) — inferimos params e derivamos scaler/feature_cols do dataset
    if isinstance(ckpt, Mapping):
        state_dict = ckpt
        feature_dim, hidden_size, num_layers = _infer_lstm_params_from_state_dict(state_dict)

        # dropout não influencia o carregamento de pesos e não é aplicado em eval() dentro do nn.LSTM
        dropout = 0.0

        model = LSTMRegressor(
            input_size=feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        ).to(DEVICE)
        model.load_state_dict(state_dict)
        model.eval()

        feature_cols, scaler_mean, scaler_scale = _derive_feature_cols_and_scaler_from_train(feature_dim)
        seq_len = _load_seq_len_hint()

        metadata = {
            "seq_len": seq_len,
            "feature_dim": feature_dim,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "feature_cols": feature_cols,
            "scaler_mean": scaler_mean,
            "scaler_scale": scaler_scale,
        }

        return model, metadata

    raise ValueError(
        f"Formato de checkpoint não suportado: {type(ckpt)}. "
        "Esperado um dict com 'model_state_dict' ou um state_dict (Mapping) de tensores."
    )


# Carrega no import (startup da API)
model, metadata = load_model()


def validate_sequence(sequence: list, normalized: bool) -> np.ndarray:
    """
    Valida shape e tipos. Retorna array numpy float32 com shape (seq_len, feature_dim).
    """
    if not isinstance(sequence, list) or len(sequence) == 0:
        raise ValueError("`sequence` deve ser uma lista não-vazia de timesteps.")

    expected_seq_len = metadata["seq_len"]
    expected_feature_dim = metadata["feature_dim"]

    # Alguns checkpoints antigos podem não carregar seq_len; nesse caso aceitamos qualquer comprimento.
    if expected_seq_len is not None and len(sequence) != expected_seq_len:
        raise ValueError(
            f"Sequência tem {len(sequence)} timesteps, mas o modelo espera {expected_seq_len}."
        )

    arr = np.array(sequence, dtype=np.float32)

    if arr.ndim != 2:
        raise ValueError(f"`sequence` deve ser 2D (seq_len x feature_dim), recebido shape {arr.shape}.")

    if arr.shape[1] != expected_feature_dim:
        raise ValueError(
            f"Cada timestep deve ter {expected_feature_dim} features, recebido {arr.shape[1]}."
        )

    return arr


def predict(sequence: list, normalized: bool = False) -> float:
    """
    Faz a predição de RUL.

    Args:
        sequence: lista de listas com shape (seq_len, feature_dim).
        normalized: se False (default), aplica o scaler salvo no checkpoint.
                    Se True, assume que os dados já estão normalizados.
    """
    arr = validate_sequence(sequence, normalized=normalized)

    # Aplica normalização se necessário (esse é o bug que estava silencioso!)
    if not normalized:
        arr = (arr - metadata["scaler_mean"]) / metadata["scaler_scale"]

    x = torch.from_numpy(arr).unsqueeze(0).to(DEVICE)  # (1, seq_len, feature_dim)

    with torch.no_grad():
        pred = model(x).cpu().numpy().item()

    return float(pred)