import pandas as pd
from pathlib import Path

test_file = Path(r"C:\Users\vish8\OneDrive\Documentos\GitHub\Ia-Systems\turbofan-rul-prediction\src\data\processed\test.parquet")

df = pd.read_parquet(test_file)

drop_columns = ["unit_id", "cycle", "rul"]
x = df.drop(columns=drop_columns, errors="ignore")

print(x.iloc[0].to_dict())