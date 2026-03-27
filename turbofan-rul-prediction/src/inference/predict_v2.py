'''
Atualizado para pegar o melhor modelo 
'''

from __future__ import annotations

import argparse
import json

import joblib
import pandas as pd

from src.core.settings import BEST_MODEL_FILE, FEATURE_TEST_FILE, PREDICTIONS_FILE, ensure_directories, DROP_COLUMNS

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type = str, default = str(FEATURE_TEST_FILE))
    parser.add_argument('--output', type = str, default = str(PREDICTIONS_FILE))
    args = parser.parse_args()
    ensure_directories()
    
    model = joblib.load(BEST_MODEL_FILE)
    df = pd.read_parquet(args.input)
    X = df.drop(columns=DROP_COLUMNS, errors="ignore")
    preds = model.predict(X)
    
    result = df[['unit_id', 'cycle', 'rul']].copy()
    result['predict_rul'] = preds
    result.to_parquet(args.output, index=False)
    print(json.dumps({"rows": len(result), "output": args.output}, indent=2))

if __name__ == '__main__':
    main()
    