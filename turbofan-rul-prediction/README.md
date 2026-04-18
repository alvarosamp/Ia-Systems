# Turbofan RUL Prediction — Production-Grade MLOps System

![CI](https://github.com/alvarosamp/Ia-Systems/actions/workflows/ci.yml/badge.svg?branch=main)
![Python](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**Live API:** [https://ia-systems.onrender.com/docs](https://ia-systems.onrender.com/docs)

A production-ready ML system for predicting the Remaining Useful Life (RUL) of aircraft turbofan engines. Built with modern MLOps practices: automated pipelines, experiment tracking, model registry with promotion gates, data validation, containerized serving, and CI/CD.

---

## The problem

Aircraft engines degrade over time due to operational stress and environmental conditions. Predicting how many operating cycles remain before failure enables **predictive maintenance** — scheduling repairs before breakdowns occur, reducing cost and increasing safety.

This project uses the NASA C-MAPSS FD001 dataset (run-to-failure sensor data from 100 turbofan engines) to train an LSTM model that predicts RUL from multivariate time series.

**Key results:**

| Metric | Value |
|--------|-------|
| Test RMSE | 16.28 |
| Test MAE | 11.48 |
| Test R² | 0.71 |

These results are within the expected range for FD001 in the literature (RMSE 11-18 for LSTM-based approaches).

---

## Live demo

The API is deployed on Render. Try it:

```bash
# Health check
curl https://ia-systems.onrender.com/health

# Model info
curl https://ia-systems.onrender.com/model-info

# Interactive docs (Swagger UI)
# Open in browser: https://ia-systems.onrender.com/docs
```

> **Note:** Free tier — first request after 15min of inactivity takes ~30s to wake up.

---

## Architecture

```
NASA C-MAPSS (.txt)
    │
    ▼
prepare.py ─── RUL clipping (125) + Pandera validation
    │
    ▼
build_features.py ─── Rolling stats, diffs + Pandera validation
    │
    ├──────────────────────────┐
    ▼                          ▼
train_lstm.py              train.py (XGBoost baseline)
(Hydra + MLflow)           (Hydra + MLflow)
    │                          │
    ▼                          ▼
MLflow Tracking ◄──────────────┘
(params, metrics/epoch, checkpoint+scaler)
    │
    ▼
register_model.py ─── MLflow Model Registry (v1, v2, ...)
    │
    ▼
promote_model.py ─── Quality gate (RMSE ≥0.5 improvement, R² ≤0.02 regression)
    │
    ▼
FastAPI + Docker ─── Loads checkpoint with scaler, serves /predict
    │
    ├── pytest (33 tests, 73% coverage) ──► GitHub Actions CI
    │
    ▼
Render (live deploy, auto-deploy on push)
```

---

## Tech stack

| Category | Tools |
|----------|-------|
| ML / DL | PyTorch (LSTM), scikit-learn, XGBoost |
| MLOps | MLflow, Hydra, Optuna, Prefect |
| Data validation | Pandera |
| API | FastAPI, Uvicorn |
| Infra | Docker, GitHub Actions, Render |
| Testing | pytest, pytest-cov, ruff |

---

## Project structure

```
turbofan-rul-prediction/
├── configs/
│   ├── config.yaml              # Global config (paths, MLflow, Optuna)
│   └── model/
│       ├── lstm.yaml             # LSTM hyperparameters (from Optuna)
│       ├── xgboost.yaml          # XGBoost baseline params
│       └── random_forest.yaml    # RF baseline params
├── src/
│   ├── core/
│   │   ├── settings.py           # Single source of truth for all paths
│   │   └── schemas.py            # Pandera data contracts
│   ├── data/
│   │   └── prepare.py            # Raw data → Parquet + RUL clipping
│   ├── features/
│   │   └── build_features.py     # Rolling stats, diffs (no leakage!)
│   ├── dl/
│   │   ├── models.py             # LSTMRegressor architecture
│   │   ├── datasets.py           # Sequences, scaler, splits
│   │   ├── train_lstm.py         # Hydra + MLflow training loop
│   │   └── optimize_lstm.py      # Optuna + Hydra HPO
│   ├── training/
│   │   ├── train.py              # Sklearn baselines (Hydra + MLflow)
│   │   └── evaluate.py           # RMSE, MAE, R²
│   ├── inference/
│   │   └── predict.py            # Batch prediction
│   ├── serving/
│   │   ├── api.py                # FastAPI endpoints
│   │   ├── inference.py          # Model loading + scaler from checkpoint
│   │   └── schema.py             # Request/response Pydantic models
│   ├── mlops/
│   │   ├── register_model.py     # MLflow Model Registry
│   │   └── promote_model.py      # Automated quality gate
│   ├── monitoring/
│   │   └── drift.py              # Data drift detection
│   └── orchestration/
│       └── flow.py               # Prefect pipeline
├── tests/
│   ├── unit/
│   │   ├── test_datasets.py      # Sequence creation, split leakage
│   │   ├── test_features_no_leakage.py  # Guards against cycle_ratio bug
│   │   └── test_schemas.py       # Pandera validation
│   └── integration/
│       └── test_api.py           # FastAPI TestClient
├── artifacts/model/              # Trained model checkpoint
├── Dockerfile                    # Hardened (non-root, healthcheck)
├── Makefile                      # Standardized commands
├── render.yaml                   # Render deploy config
└── requirements.txt
```

---

## How to run

### Prerequisites

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

### Full pipeline

```bash
make pipeline  # prepare → features → train → register
```

### Individual steps

```bash
make prepare     # Raw data → Parquet (with RUL clipping at 125)
make features    # Feature engineering (rolling stats, diffs)
make optimize    # Optuna hyperparameter search
make train       # Train LSTM with best params
make test        # Run pytest suite
make api         # Start local API server
make register    # Register model in MLflow Registry
make promote     # Run quality gate (Staging → Production)
```

### Hydra overrides

```bash
# Train with custom params
python -m src.dl.train_lstm model.params.lr=0.01 model.training.epochs=50

# Optimize with more trials
python -m src.dl.optimize_lstm optuna.n_trials=50
```

### Docker

```bash
docker build -t turbofan-rul .
docker run --rm -p 8000:8000 turbofan-rul
# API available at http://localhost:8000/docs
```

---

## MLflow experiment tracking

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Open http://127.0.0.1:5000
```

Every training run logs: all hyperparameters, per-epoch metrics (train_loss, val_rmse, val_mae, val_r2, lr), final test metrics, the model checkpoint (with scaler embedded), and training history.

---

## Model registry and quality gate

The promotion gate prevents deploying a worse model to production:

```bash
python -m src.mlops.register_model   # → v2 in Staging
python -m src.mlops.promote_model    # → compares v2 vs v1 (Production)
```

**Gate criteria (both must pass):**
- RMSE must improve by at least 0.5
- R² must not regress by more than 0.02

Example of the gate correctly rejecting a candidate:

```
Candidato  (v2): rmse=16.2130 r2=0.7134
Production (v1): rmse=16.2812 r2=0.7110
Gate: rmse_improvement=+0.0681 (precisa >= 0.5)  ❌
Gate: r2_regression=-0.0024    (precisa <= 0.02)  ✅
❌ Candidato v2 REPROVADO. Production (v1) mantido.
```

The improvement of 0.068 RMSE is within run-to-run noise — the gate correctly identifies this and protects production from unnecessary model churn.

---

## Data leakage story

During development, the LSTM initially reported a validation RMSE of 5.1 — well below the state-of-the-art for FD001 (~11-13). This was suspicious.

Investigation revealed that `cycle_ratio = cycle / max(cycle)` was **silent data leakage**: in train, `max(cycle)` is the failure cycle (essentially the normalized target); in test, it's just the last observed cycle (a completely different quantity). The model learned to rely on this "magic feature" and collapsed on test data (RMSE 54.9, R² -0.12).

After removing the leaky features and applying RUL clipping at 125 (standard in C-MAPSS literature since Heimes 2008), the model achieved honest metrics:

| | Before fix | After fix |
|---|---|---|
| Val RMSE (Optuna) | 5.10 | 12.96 |
| Test RMSE | 54.90 | 16.28 |
| Test R² | -0.12 | 0.71 |

A regression test (`test_features_no_leakage.py`) now permanently guards against this bug.

---

## API endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check (model loaded?) |
| GET | `/model-info` | Model architecture and config |
| POST | `/predict` | Predict RUL from sensor sequence |
| GET | `/docs` | Interactive Swagger UI |

### Predict request

```json
{
  "sequence": [[0.1, 0.2, ...], [0.1, 0.3, ...], ...],
  "normalized": false
}
```

- `sequence`: 2D array with shape `(seq_len, feature_dim)` — raw sensor readings
- `normalized`: if `false` (default), the API applies the scaler from the training checkpoint automatically. If `true`, assumes data is already normalized.

The scaler (mean and std from training) is embedded in the model checkpoint — no external preprocessing needed.

---

## Tests

```bash
pytest --cov=src --cov-report=term-missing
```

**33 tests, 73% coverage.** Critical modules (schemas, datasets, features, evaluate) have 90%+ coverage. Entry points (train scripts) are validated via integration tests.

Key test: `test_features_no_leakage.py` ensures that `cycle_ratio`, `cycle_left_proxy`, and `cycle_squared` can never be reintroduced — these were the source of the data leakage bug.

---

## Future work

- Prometheus + Grafana for API observability (latency, request count, prediction distribution)
- Evidently AI for automated data/model drift detection
- SHAP explainability endpoint (`/explain`)
- MLflow `pyfunc` packaging for framework-agnostic serving
- Additional C-MAPSS datasets (FD002, FD003, FD004)

---

## Author

**Alvaro Sampaio** — [GitHub](https://github.com/alvarosamp)
