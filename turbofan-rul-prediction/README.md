# Turbofan RUL Prediction — Production-Grade ML System ✈️

Design and implementation of a production-oriented machine learning system for predicting Remaining Useful Life (RUL) of turbofan engines using the NASA C-MAPSS dataset.

---

## 🧠 Problem Context

Aircraft engines degrade over time due to operational stress and environmental conditions.

The objective is to predict:

Remaining Useful Life (RUL) — number of cycles before engine failure.

Why this matters:
- Prevent unexpected failures
- Enable predictive maintenance
- Reduce operational costs
- Increase safety

---

## 🏗️ System Architecture

Raw Data → Data Preparation → Feature Engineering → Training → MLflow → Model Selection → Inference → Monitoring

---

## ⚙️ Tech Stack

ML:
- scikit-learn
- XGBoost

MLOps:
- MLflow
- Hydra
- Optuna

Backend:
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
