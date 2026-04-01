import numpy as np
from torch.utils.data import Dataset
from src.core.settings import DROP_COLUMNS
from sklearn.preprocessing import StandardScaler
import pandas as pd

def split_train_val_by_unit(df: pd.DataFrame, val_ratio: float = 0.2):
    unit_ids = sorted(df["unit_id"].unique())
    split_idx = int(len(unit_ids) * (1 - val_ratio))

    train_units = unit_ids[:split_idx]
    val_units = unit_ids[split_idx:]

    train_df = df[df["unit_id"].isin(train_units)].copy()
    val_df = df[df["unit_id"].isin(val_units)].copy()

    return train_df, val_df

def fit_feature_scaler(train_df: pd.DataFrame):
    feature_cols = [c for c in train_df.columns if c not in DROP_COLUMNS]
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols])
    return scaler, feature_cols

def apply_feature_scaler(df: pd.DataFrame, scaler: StandardScaler, feature_cols: list[str]) -> pd.DataFrame:
    df_scaled = df.copy()
    df_scaled[feature_cols] = scaler.transform(df[feature_cols])
    return df_scaled

def create_sequences(df: pd.DataFrame, seq_len: int = 30):
    X, y = [], []

    feature_cols = [c for c in df.columns if c not in DROP_COLUMNS]

    for unit_id in df["unit_id"].unique():
        unit_df = df[df["unit_id"] == unit_id].sort_values("cycle")

        features = unit_df[feature_cols].values
        targets = unit_df["rul"].values

        if len(unit_df) <= seq_len:
            continue

        for i in range(len(unit_df) - seq_len):
            X.append(features[i:i + seq_len])
            y.append(targets[i + seq_len])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


class TurbofanDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]