from __future__ import annotations

from prometheus_client import Counter, Histogram, Gauge


# Contador de predições totais
PREDICTION_COUNT = Counter(
    "turbofan_predictions_total",
    "Total number of RUL predictions made",
    ["status"],  # "success" ou "error"
)

# Histograma de latência do /predict
PREDICTION_LATENCY = Histogram(
    "turbofan_prediction_latency_seconds",
    "Latency of /predict endpoint in seconds",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)

# Histograma da distribuição de RUL previsto
PREDICTED_RUL_DISTRIBUTION = Histogram(
    "turbofan_predicted_rul",
    "Distribution of predicted RUL values",
    buckets=[0, 10, 20, 30, 50, 75, 100, 125, 150],
)

# Gauge do modelo em uso
MODEL_INFO = Gauge(
    "turbofan_model_info",
    "Model metadata",
    ["seq_len", "hidden_size", "num_layers"],
)