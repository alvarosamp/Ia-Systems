import pandas as pd
from pathlib import Path

# Caminhos dos arquivos de features
train_path = Path("src/data/processed/train_features.parquet")
test_path = Path("src/data/processed/test_features.parquet")

print("==== PRIMEIRAS LINHAS DO TREINO ====")
df_train = pd.read_parquet(train_path)
print(df_train.head())
print("\nResumo estatístico do treino:")
print(df_train.describe())

print("\n==== PRIMEIRAS LINHAS DO TESTE ====")
df_test = pd.read_parquet(test_path)
print(df_test.head())
print("\nResumo estatístico do teste:")
print(df_test.describe())

print("\nColunas disponíveis:")
print(df_train.columns)

print("\nValores únicos em 'rul' (target) do treino:")
print(df_train['rul'].unique()[:20])
print(f"Total de valores únicos: {df_train['rul'].nunique()}")
