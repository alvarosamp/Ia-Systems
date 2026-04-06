from pydantic import BaseModel, Field
from typing import List


class SequenceRequest(BaseModel):
    sequence: List[List[float]] = Field(
        ...,
        description="Sequence of timesteps with shape (seq_len x num_features)"
    )


class PredictionResponse(BaseModel):
    predicted_rul: float