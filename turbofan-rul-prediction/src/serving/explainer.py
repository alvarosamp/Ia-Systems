from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
from captum.attr import IntegratedGradients

from src.serving.inference import model, metadata, DEVICE


def _build_baseline(seq_len: int, feature_dim: int) -> torch.Tensor:
    """
    Baseline pra Integrated Gradients: sequência de zeros (normalizada).

    Em termos do scaler, zero = a média do treino. Isso significa
    "um input onde todos os sensores estão no valor médio" — ou seja,
    nenhum sinal de degradação. Faz sentido fisicamente porque estamos
    perguntando: "o que nessa sequência, comparado com uma turbina média,
    fez o modelo prever esse RUL?"
    """
    return torch.zeros(1, seq_len, feature_dim, dtype=torch.float32).to(DEVICE)


def _get_feature_names() -> List[str]:
    """Retorna a lista de nomes das features, na ordem que o modelo espera."""
    return metadata["feature_cols"]


def explain_prediction(
    sequence: List[List[float]],
    normalized: bool = False,
    top_k: int = 10,
    n_steps: int = 50,
) -> Dict:
    """
    Explica uma predição de RUL usando Integrated Gradients.

    Como funciona:
    1. Pega a sequência de input (seq_len timesteps x feature_dim features)
    2. Cria uma baseline (todos zeros = "turbina média")
    3. Interpola do baseline até o input em n_steps passos
    4. Calcula o gradiente do output em relação ao input em cada passo
    5. Integra (soma) esses gradientes
    6. O resultado é a "importância" de cada feature em cada timestep

    O output agrega por feature (soma absoluta ao longo dos timesteps)
    e retorna as top_k features mais importantes.

    Args:
        sequence: lista de listas (seq_len x feature_dim), dados brutos ou normalizados
        normalized: se False, aplica o scaler do checkpoint
        top_k: número de features mais importantes a retornar
        n_steps: passos de interpolação (mais = mais preciso, mais lento)

    Returns:
        dict com predicted_rul, top_features (nome, importance, direction),
        e timestep_importance (quais timesteps mais influenciaram)
    """
    arr = np.array(sequence, dtype=np.float32)

    # Aplica normalização se necessário (mesmo scaler da predição)
    if not normalized:
        arr = (arr - metadata["scaler_mean"]) / metadata["scaler_scale"]

    input_tensor = torch.from_numpy(arr).unsqueeze(0).to(DEVICE)  # (1, seq_len, feature_dim)
    input_tensor.requires_grad_(True)

    baseline = _build_baseline(
        seq_len=metadata["seq_len"],
        feature_dim=metadata["feature_dim"],
    )

    # Faz a predição primeiro
    model.eval()
    with torch.no_grad():
        predicted_rul = model(input_tensor).cpu().numpy().item()

    # Integrated Gradients
    # Em GPU, o cuDNN RNN não permite backward em eval mode.
    # Para manter a explicação determinística (sem dropout), mantemos eval()
    # e desabilitamos cuDNN apenas durante a atribuição.
    with torch.backends.cudnn.flags(enabled=False):
        ig = IntegratedGradients(model)
        attributions = ig.attribute(
            input_tensor,
            baselines=baseline,
            n_steps=n_steps,
            return_convergence_delta=False,
        )

    # attributions shape: (1, seq_len, feature_dim)
    attr_np = attributions.squeeze(0).detach().cpu().numpy()  # (seq_len, feature_dim)

    # Agrega por feature: soma dos valores absolutos ao longo dos timesteps
    feature_importance = np.abs(attr_np).sum(axis=0)  # (feature_dim,)

    # Direção média por feature (positivo = aumenta RUL, negativo = diminui)
    feature_direction = attr_np.sum(axis=0)  # (feature_dim,)

    # Agrega por timestep: soma dos valores absolutos ao longo das features
    timestep_importance = np.abs(attr_np).sum(axis=1)  # (seq_len,)

    # Normaliza pra percentual
    total = feature_importance.sum()
    if total > 0:
        feature_importance_pct = feature_importance / total
    else:
        feature_importance_pct = feature_importance

    # Top-K features
    feature_names = _get_feature_names()
    top_indices = np.argsort(feature_importance_pct)[::-1][:top_k]

    top_features = []
    for idx in top_indices:
        direction = "increases RUL" if feature_direction[idx] > 0 else "decreases RUL"
        top_features.append({
            "feature": feature_names[idx],
            "importance": round(float(feature_importance_pct[idx]) * 100, 2),
            "direction": direction,
            "raw_attribution": round(float(feature_direction[idx]), 4),
        })

    # Timestep importance (normalizado)
    ts_total = timestep_importance.sum()
    if ts_total > 0:
        ts_importance_pct = (timestep_importance / ts_total * 100).tolist()
    else:
        ts_importance_pct = timestep_importance.tolist()

    return {
        "predicted_rul": float(predicted_rul),
        "top_features": top_features,
        "timestep_importance": [round(v, 2) for v in ts_importance_pct],
        "explanation_method": "Integrated Gradients (Captum)",
        "n_steps": n_steps,
        "baseline": "zero (normalized mean of training data)",
    }