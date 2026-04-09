from __future__ import annotations

from typing import List
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