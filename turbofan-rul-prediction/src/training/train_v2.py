from __future__ import annotations

import json

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.core.settings import (
    FEATURE_TRAIN_FILE,
    FEATURE_TEST_FILE,
    BEST_MODEL_FILE,
    METRICS_FILE,
    ensure_directories,
    DROP_COLUMNS,
)
from src.training.evaluate import evaluate_regression


def load_data():
    train_df = pd.read_parquet(FEATURE_TRAIN_FILE)
    test_df = pd.read_parquet(FEATURE_TEST_FILE)
    return train_df, test_df

def split_Xy(df: pd.DataFrame):
    X = df.drop(columns=DROP_COLUMNS, errors="ignore")
    y = df["rul"]
    return X, y

def build_random_forest():
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=100,
                    max_depth=12,
                    random_state=42,
                    n_jobs=-1,
                    min_samples_split = 4,
                    min_samples_leaf= 2
                ),
            ),
        ]
    )

def build_xgboost():
    return XGBRegressor(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1,
        objective="reg:squarederror",
    )
    
def train_and_evaluate(model_name: str, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    metrics = evaluate_regression(y_test, preds)
    return model, metrics

def main():
    ensure_directories()
    train_df, test_df = load_data()
    X_train, y_train = split_Xy(train_df)
    X_test, y_test = split_Xy(test_df)
    
    candidates = {
        "random_forest": build_random_forest(),
        "xgboost": build_xgboost(),
    }
    
    results = {}
    trained_models = {}
    mlflow.set_experiment("turbofan_rul_prediction-v2")
    with mlflow.start_run(run_name="model_comparison_v2"):
        for model_name, model in candidates.items():
            trained_model, metrics = train_and_evaluate(
                model_name, model, X_train, y_train, X_test, y_test
            )
            trained_models[model_name] = trained_model
            results[model_name] = metrics
            mlflow.log_metrics({f"{model_name}_{k}": v for k, v in metrics.items()})
            
        # Salvar os resultados e o melhor modelo
        best_model_name = min(results, key=lambda k: results[k]["rmse"])
        best_model = trained_models[best_model_name]
        
        joblib.dump(best_model, BEST_MODEL_FILE)
        
        summary = {
            'best_model': best_model_name,
            'metrics': results,
        }
        
        with open(METRICS_FILE, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
            
        mlflow.log_param("best_model", best_model_name)
        mlflow.log_artifact(str(BEST_MODEL_FILE))
        mlflow.log_artifact(str(METRICS_FILE))
        
        print(json.dumps(summary, indent=2))
        
if __name__ == "__main__":
    main()
    