# Turbofan RUL Prediction — Production-Grade ML System ✈️

Sistema de machine learning para previsão do Remaining Useful Life (RUL) de turbinas aeronáuticas, com arquitetura modular, rastreamento de experimentos, automação de pipeline, monitoramento e deploy via API.

---

## 🧠 Contexto do Problema

Motores de aeronaves degradam com o tempo devido a estresse operacional e condições ambientais. O objetivo é prever o RUL — ciclos restantes até a falha — para:
- Prevenir falhas inesperadas
- Permitir manutenção preditiva
- Reduzir custos operacionais
- Aumentar a segurança

---

## 🏗️ Arquitetura do Sistema

**Pipeline:**  
Raw Data → Data Preparation → Feature Engineering → Training → MLflow/Optuna → Model Selection → Inference → Monitoring

- **Automação:** Orquestração com Prefect (`src/orchestration/flow.py`)
- **Gerenciamento de Experimentos:** MLflow, Optuna
- **Monitoramento:** Evidently, scripts customizados
- **Deploy:** FastAPI

---

## ⚙️ Stack Tecnológico

- **ML:** scikit-learn, XGBoost
- **MLOps:** MLflow, Hydra, Optuna, DVC, Prefect, Evidently
- **Backend:** FastAPI, Uvicorn
- **Dados:** Parquet
- **Infra:** Docker

---

## 📂 Estrutura de Pastas

- **src/data/**: Preparação dos dados brutos e processamento inicial
- **src/features/**: Engenharia de features (rolling mean/std, normalização, remoção de constantes)
- **src/training/**: Treinamento, avaliação, otimização de hiperparâmetros
- **src/inference/**: Inferência e geração de predições
- **src/api/**: API FastAPI para servir o modelo
- **src/monitoring/**: Scripts de monitoramento e detecção de drift
- **artifacts/**: Modelos, métricas, resultados de Optuna
- **configs/**: Configurações Hydra para modelos, treinamento e experimentos
- **data/**: Dados brutos e processados
- **outputs/**: Resultados de execuções

---

## 🔄 Pipeline Automatizado

- **DVC:** Define as etapas do pipeline (prepare, build_features, train, predict) em `dvc.yaml`
- **Prefect:** Orquestração programática do fluxo de ponta a ponta

---

## 🧩 Configuração e Customização

- **Hydra:** Permite trocar modelos e parâmetros facilmente via linha de comando ou arquivos YAML
- **Exemplo de configuração (`configs/model/xgboost.yaml`):**
	```yaml
	name: xgboost
	params:
		n_estimators: 300
		max_depth: 8
		learning_rate: 0.05
		...
	```
- **Configuração global (`configs/config.yaml`):** Define caminhos, nome do experimento, parâmetros de Optuna, etc.

---

## 🚀 Como Executar

1. **Instale as dependências:**
	 ```bash
	 pip install -r requirements.txt
	 ```

2. **Execute o pipeline completo:**
	 ```bash
	 python -m src.orchestration.flow
	 ```
	 Ou etapas individuais:
	 ```bash
	 python -m src.data.prepare
	 python -m src.features.build_features
	 python -m src.training.train
	 python -m src.inference.predict
	 ```

	 Otimização (Optuna) do modelo LSTM:
	 ```bash
	 python -m src.dl.optimize_lstm
	 ```
	 **Nota:** ao usar `python -m`, não adicione o sufixo `.py` (ex.: use `src.dl.optimize_lstm`, não `src.dl.optimize_lstm.py`).
	 Se seu `python` não estiver apontando para o venv correto, execute com o interpretador do venv explicitamente.

3. **Rastreie experimentos:**
	 ```bash
	 mlflow ui
	 ```

4. **API para inferência:**
	 ```bash
	 uvicorn src.api.main:app --reload
	 ```

5. **Monitoramento de drift:**
	 ```bash
	 python -m src.monitoring.drift
	 ```

---

## 🧪 Testes

- Testes unitários em `tests/unit/`
- Rodar com:
	```bash
	pytest
	```

---

## 📊 Métricas

- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² (Coeficiente de Determinação)

---

## 🛠️ Extensibilidade

- **Adicionar novos modelos:** Crie um novo arquivo YAML em `configs/model/` e implemente o script correspondente em `src/training/`
- **Novas features:** Edite `src/features/build_features.py`
- **Monitoramento:** Expanda `src/monitoring/` ou integre Evidently

---

## 🗂️ Dados

- **Origem:** NASA C-MAPSS (em `src/data/raw/`)
- **Processados:** Parquet em `src/data/processed/`
- **Features:** Parquet em `src/data/processed/`
- **Predições:** Salvas em `artifacts/model/predictions.parquet`

---

## 🐳 Docker

- **Build:**
	```bash
	docker build -t turbofan-rul .
	```
- **Run:**
	```bash
	docker run --rm -p 8000:8000 turbofan-rul
	```

---

## 🔥 Futuro

- Integração CI/CD
- Deploy em nuvem
- Monitoramento contínuo
- Mais modelos e explainability

---

## 💼 Resumo

Este projeto demonstra um pipeline de ML robusto, modular, rastreável e pronto para produção, com práticas modernas de MLOps e fácil extensibilidade.

---
- FastAPI

Data:
- Parquet

---

## 🔬 Feature Engineering

- Rolling mean (window=5)
- Rolling std
- Cycle ratio normalization
- Removal of constant features

---

## 🤖 Models

- Random Forest (baseline)
- XGBoost (advanced)

---

## ⚡ Experiment Tracking

Use MLflow:

mlflow ui

---

## 🚀 How to Run

pip install -r requirements.txt

python -m src.data.prepare
python -m src.features.build_features
python -m src.training.train
python -m src.inference.predict

---

## 📊 Metrics

- RMSE
- MAE
- R²

---

## ⚙️ Hydra Usage

python -m src.training.train model=xgboost

---

## 🔥 Future Work

- DVC
- Prefect
- Monitoring (Evidently)
- Docker
- CI/CD

---

## 💼 Summary

This project demonstrates a production-ready ML pipeline with strong MLOps foundations, modular design, and scalability for real-world applications.
