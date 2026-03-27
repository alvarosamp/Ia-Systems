from __future__ import annotations

import json

import mlflow
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.core.settings import FEATURE_TRAIN_FILE, OPTUNA_STUDY_FILE,DROP_COLUMNS, ensure_directories


def load_training_data():
    df = pd.read_parquet(FEATURE_TRAIN_FILE)
    X = df.drop(columns=DROP_COLUMNS, errors="ignore")
    y = df['rul']
    return X, y

def rf_objective(trial, X,y):
    model  = Pipeline(
        steps =[
            ('scaler', StandardScaler()),
            ('regressor', RandomForestRegressor(
                n_estimators = trial.suggest_int('n_estimators', 100, 500),
                max_depth = trial.suggest_int('max_depth', 5, 20),
                min_samples_split = trial.suggest_int('min_samples_split', 2, 10),
                min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 5),
                random_state = 42,
                n_jobs=-1
            ))
        ]
    )
    cv = KFold(n_splits = 5, shuffle = True, random_state = 42)
    score = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1).mean()
    return -score

def xbg_objetive(trial, X, y):
    model = XGBRegressor(
        n_estimators=trial.suggest_int("n_estimators", 100, 500),
        max_depth=trial.suggest_int("max_depth", 3, 12),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        random_state=42,
        n_jobs=-1,
        objective="reg:squarederror",
    )
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    score = cross_val_score(model, X, y, cv=cv, scoring="neg_mean_squared_error", n_jobs=-1).mean()
    return -score


def main():
    ensure_directories()
    X , y = load_training_data()
    mlflow.set_experiment("turbofan_rul_prediction")
    results = {}
    
    with mlflow.start_run(run_name = 'Optuna_rf_xgb'):
        #RF
        rf_study = optuna.create_study(direction = 'minimize')
        rf_study.optimize(lambda trial: rf_objective(trial, X, y), n_trials = 50)
        results['rf'] = {
            'best_value_rmse' : rf_study.best_value,
            'best_params' : rf_study.best_params,
        }
        #XGB
        xgb_study = optuna.create_study(direction = 'minimize')
        xgb_study.optimize(lambda trial: xbg_objetive(trial, X, y), n_trials = 50)
        results['xgb'] = {
            'best_value_rmse' : xgb_study.best_value,
            'best_params' : xgb_study.best_params
        }
        mlflow.log_dict(results, 'optuna_results.json')
        with open(OPTUNA_STUDY_FILE, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        
        print(json.dumps(results, indent= 2))
        
if __name__ == "__main__":
    main()
        
    
    


