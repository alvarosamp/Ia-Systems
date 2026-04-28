from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class SequenceRequest(BaseModel):
    sequence: List[List[float]] = Field(
        ...,
        description="Sequência de timesteps com shape (seq_len, feature_dim). Dados brutos por padrão.",
    )
    normalized: bool = Field(
        default=False,
        description="Se True, assume que `sequence` já está normalizada com o scaler do treino.",
    )


class PredictionResponse(BaseModel):
    predicted_rul: float


class ExplainRequest(BaseModel):
    sequence: List[List[float]] = Field(
        ...,
        description="Sequência de timesteps com shape (seq_len, feature_dim).",
    )
    normalized: bool = Field(
        default=False,
        description="Se True, assume dados já normalizados.",
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Número de features mais importantes a retornar.",
    )


class FeatureAttribution(BaseModel):
    feature: str
    importance: float = Field(description="Importância relativa em percentual (0-100)")
    direction: str = Field(description="'increases RUL' ou 'decreases RUL'")
    raw_attribution: float = Field(description="Valor bruto da atribuição (positivo = aumenta RUL)")


class ExplainResponse(BaseModel):
    predicted_rul: float
    top_features: List[FeatureAttribution]
    timestep_importance: List[float] = Field(
        description="Importância de cada timestep em percentual. Últimos timesteps geralmente pesam mais."
    )
    explanation_method: str
    n_steps: int
    baseline: str