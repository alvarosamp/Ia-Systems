import pandas as pd 
import numpy as np
from torch.utils.data import Dataset
from src.core.settings import DROP_COLUMNS

def create_sequences(df, seq_len = 30):
    X, y = [], []
    for unit_id in df['unit_id'].unique():
        unit_id = df[df['unit_id'] == unit_id].sort_values('cycle')
        features = unit_id.drop(columns = DROP_COLUMNS, errors="ignore").values
        targets = unit_id['rul'].values
        for i in range(len(unit_id) - seq_len):
            X.append(features[i:i+seq_len])
            y.append(targets[i+seq_len])
    return np.array(X), np.array(y)

class TurboFanDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]