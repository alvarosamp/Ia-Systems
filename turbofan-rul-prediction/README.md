# Turbofan RUL Prediction

Predição de Remaining Useful Life (RUL) para motores turbofan usando o dataset NASA C-MAPSS.

## Objetivo
Construir uma pipeline de machine learning para:
- Preparar e processar dados
- Extrair features
- Treinar e otimizar modelos
- Salvar artefatos e métricas
- Realizar inferência
- Expor uma API para predição

---

## Estrutura do Projeto

```
turbofan-rul-prediction/
├── data/
│   └── raw/           # Coloque aqui os arquivos do dataset C-MAPSS
│   └── processed/     # Dados processados e features
├── src/
│   ├── core/          # Configurações e utilitários
│   ├── data/          # Scripts de preparação de dados
│   ├── features/      # Extração de features
│   ├── training/      # Treinamento, avaliação e otimização
│   ├── inference/     # Scripts de inferência
│   └── api/           # API FastAPI
├── artifacts/         # Modelos, métricas, relatórios
├── requirements.txt   # Dependências
├── pyproject.toml     # Configurações de formatação e testes
└── README.md
```

---

## Dataset esperado
Coloque estes arquivos em `src/data/raw/archive/CMaps/`:
- `train_FD001.txt`
- `test_FD001.txt`
- `RUL_FD001.txt`

---

## Instalação

1. Crie e ative um ambiente virtual (opcional, mas recomendado):
	```bash
	python -m venv .venv
	source .venv/bin/activate  # Linux/Mac
	.venv\Scripts\activate    # Windows
	```
2. Instale as dependências:
	```bash
	pip install -r requirements.txt
	```

---

## Pipeline de Execução

### 1. Preparar os dados brutos
Processa os arquivos brutos e gera os dados processados.
```bash
python -m src.data.prepare
```

### 2. Gerar features
Extrai features e salva arquivos Parquet para treino e teste.
```bash
python -m src.features.build_features
```

### 3. Treinar modelos
Treina modelos Random Forest e XGBoost, salva o melhor modelo e métricas.
```bash
python -m src.training.train_v2
```

### 4. Otimizar hiperparâmetros (opcional)
Executa busca de hiperparâmetros com Optuna.
```bash
python -m src.training.optimize
```

### 5. Inferência
Gera predições usando o melhor modelo treinado.
```bash
python -m src.inference.predict_v2
```

### 6. Executar API (opcional)
Exponha o modelo via FastAPI:
```bash
uvicorn src.api.main:app --reload
```

---

## Principais Scripts

- **src/data/prepare.py**: Processa os dados brutos do C-MAPSS.
- **src/features/build_features.py**: Extrai features e salva em Parquet.
- **src/training/train_v2.py**: Treina modelos, salva artefatos e métricas.
- **src/training/optimize.py**: Otimização de hiperparâmetros com Optuna.
- **src/inference/predict_v2.py**: Realiza predições com o melhor modelo.
- **src/api/main.py**: API para servir o modelo (FastAPI).

---

## Dependências principais

- pandas, numpy, scikit-learn, xgboost
- mlflow, optuna, joblib
- fastapi, uvicorn
- pyarrow (para Parquet)
- evidently (monitoramento)
- dvc (controle de dados)

Veja todas em `requirements.txt`.

---

## Dicas e Troubleshooting

- **Erro Parquet**: Certifique-se de que todos os arquivos `.parquet` foram gerados com `to_parquet` e lidos com `read_parquet`.
- **ImportError src**: Sempre execute os scripts a partir da raiz do projeto usando `python -m ...`.
- **Ambiente virtual**: Ative o ambiente antes de rodar os scripts.
- **MLflow**: Para visualizar os experimentos, rode `mlflow ui` na raiz do projeto.

---

## Testes

Para rodar os testes unitários:
```bash
pytest
```

---

## Autor

Projeto desenvolvido por [Seu Nome].