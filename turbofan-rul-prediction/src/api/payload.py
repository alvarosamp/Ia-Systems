from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch

ROOT_DIR = Path(r"C:\Users\vish8\OneDrive\Documentos\GitHub\Ia-Systems\turbofan-rul-prediction")
TEST_FEATURES_FILE = r"C:\Users\vish8\OneDrive\Documentos\GitHub\Ia-Systems\turbofan-rul-prediction\src\data\processed\test_features.parquet"
MODEL_FILE = r"C:\Users\vish8\OneDrive\Documentos\GitHub\Ia-Systems\turbofan-rul-prediction\src\artifacts\model\lstm_v2.pth"
OUTPUT_FILE = r"C:\Users\vish8\OneDrive\Documentos\GitHub\Ia-Systems\turbofan-rul-prediction\artifacts\reports\sample_api_payload.json"

DROP_COLUMNS = ["unit_id", "cycle", "rul"]

checkpoint = torch.load(MODEL_FILE, map_location="cpu")
feature_cols = checkpoint["feature_cols"]
seq_len = checkpoint["seq_len"]

test_df = pd.read_parquet(TEST_FEATURES_FILE).copy()

# aplica o mesmo scaler salvo no checkpoint
mean = np.array(checkpoint["scaler_mean"])
scale = np.array(checkpoint["scaler_scale"])
test_df[feature_cols] = (test_df[feature_cols] - mean) / scale

# pega uma turbina qualquer com dados suficientes
unit_id = int(test_df["unit_id"].iloc[0])
unit_df = test_df[test_df["unit_id"] == unit_id].sort_values("cycle")

# garante seq_len pontos
sequence = unit_df[feature_cols].iloc[:seq_len].values.tolist()

payload = {"sequence": sequence}

Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)

print(f"Payload salvo em: {OUTPUT_FILE}")
print(f"unit_id usado: {unit_id}")
print(f"seq_len: {seq_len}")
print(f"n_features: {len(feature_cols)}")