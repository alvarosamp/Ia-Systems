import numpy as np
from torch.utils.data import Dataset


DROP_COLUMNS = ["unit_id", "cycle", "rul"]


def create_sequences(df, seq_len=30):
    X, y = [], []

    for unit_id in df["unit_id"].unique():
        unit_df = df[df["unit_id"] == unit_id].sort_values("cycle")

        features = unit_df.drop(columns=DROP_COLUMNS).values
        targets = unit_df["rul"].values

        for i in range(len(unit_df) - seq_len):
            X.append(features[i:i + seq_len])
            y.append(targets[i + seq_len])

    return np.array(X), np.array(y)


class TurbofanDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]