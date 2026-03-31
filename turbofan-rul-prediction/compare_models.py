
import subprocess
import sys
import json
from pathlib import Path
import os

def run_model(model_name):
    """Executa o treinamento para o modelo especificado e retorna as métricas."""
    print(f"\nTreinando modelo: {model_name}")
    # Executa o script train_v2.py com o modelo desejado
    result = subprocess.run([
        sys.executable, "-m", "src.training.train_v2", f"model={model_name}"
    ], capture_output=True, text=True)
    print(result.stdout)
    # Descobre o diretório raiz do projeto (onde está este script)
    project_root = Path(__file__).parent.resolve()
    metrics_path = project_root / "artifacts" / "model" / f"metrics_{model_name}.json"
    print(f"[DEBUG] Procurando métricas em: {metrics_path}")
    if metrics_path.exists():
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)
            return metrics
        except Exception as e:
            print(f"[ERRO] Falha ao ler métricas para {model_name}: {e}")
            return None
    else:
        print(f"Métricas não encontradas para {model_name} em {metrics_path}.")
        return None

def main():
    results = {}
    for model in ["random_forest", "xgboost"]:
        metrics = run_model(model)
        results[model] = metrics

    print("\n================ COMPARAÇÃO DE MODELOS ================")
    for model, metrics in results.items():
        print(f"\nModelo: {model}")
        if metrics:
            for k, v in metrics.items():
                print(f"  {k}: {v}")
        else:
            print("  Métricas não disponíveis.")

if __name__ == "__main__":
    main()