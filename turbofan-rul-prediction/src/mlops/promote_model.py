from __future__ import annotations

import sys

import hydra
import mlflow
from mlflow.tracking import MlflowClient
from omegaconf import DictConfig

# IMPORTANTE: tem que bater com o nome usado em register_model.py
MODEL_NAME = "turbofan_rul_model"

# Thresholds de promoção
MIN_RMSE_IMPROVEMENT = 0.5  # candidato precisa melhorar RMSE em >= 0.5
MAX_R2_REGRESSION = 0.02    # R² não pode piorar mais que 0.02


def _get_latest_version_by_stage(client: MlflowClient, stage: str):
    """
    Retorna a versão mais recente com tag stage == `stage`.
    Usa v.tags direto (search_model_versions já traz as tags).
    """
    try:
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    except Exception as e:
        print(f"Erro ao buscar versões de {MODEL_NAME}: {e}")
        return None

    matching = [v for v in versions if v.tags.get("stage") == stage]
    if not matching:
        return None

    # Ordena por número de versão (decrescente) e pega a maior
    return max(matching, key=lambda v: int(v.version))


def _metric_from_tags(version, key: str) -> float:
    """Lê métrica salva como tag na versão."""
    tag = version.tags.get(key)
    if tag is None or tag == "NA":
        return float("nan")
    return float(tag)


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    client = MlflowClient()

    # Debug: lista todas as versões do modelo
    all_versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    if not all_versions:
        print(f"❌ Nenhuma versão encontrada para o modelo '{MODEL_NAME}'.")
        print("   Rode primeiro: python -m src.mlops.register_model")
        sys.exit(1)

    print(f"Versões existentes de {MODEL_NAME}:")
    for v in all_versions:
        stage = v.tags.get("stage", "None")
        rmse = v.tags.get("test_rmse", "NA")
        r2 = v.tags.get("test_r2", "NA")
        print(f"  v{v.version} | stage={stage} | rmse={rmse} | r2={r2}")
    print()

    candidate = _get_latest_version_by_stage(client, "Staging")
    if candidate is None:
        print("❌ Nenhum modelo em Staging.")
        print("   Verifique que register_model.py setou a tag 'stage'='Staging'.")
        sys.exit(1)

    current = _get_latest_version_by_stage(client, "Production")

    cand_rmse = _metric_from_tags(candidate, "test_rmse")
    cand_r2 = _metric_from_tags(candidate, "test_r2")

    print(f"Candidato  (v{candidate.version}): rmse={cand_rmse:.4f} r2={cand_r2:.4f}")

    # Primeira promoção: não existe Production ainda
    if current is None:
        print("Nenhum modelo em Production. Promovendo candidato direto.")
        client.set_model_version_tag(MODEL_NAME, candidate.version, "stage", "Production")
        print(f"✅ v{candidate.version} -> Production")
        return

    prod_rmse = _metric_from_tags(current, "test_rmse")
    prod_r2 = _metric_from_tags(current, "test_r2")
    print(f"Production (v{current.version}): rmse={prod_rmse:.4f} r2={prod_r2:.4f}")

    rmse_improvement = prod_rmse - cand_rmse
    r2_regression = prod_r2 - cand_r2

    print()
    print(f"Gate: rmse_improvement={rmse_improvement:+.4f} (precisa >= {MIN_RMSE_IMPROVEMENT})")
    print(f"Gate: r2_regression={r2_regression:+.4f}    (precisa <= {MAX_R2_REGRESSION})")
    print()

    if rmse_improvement >= MIN_RMSE_IMPROVEMENT and r2_regression <= MAX_R2_REGRESSION:
        client.set_model_version_tag(MODEL_NAME, current.version, "stage", "Archived")
        client.set_model_version_tag(MODEL_NAME, candidate.version, "stage", "Production")
        print(f"✅ v{candidate.version} PROMOVIDO para Production.")
        print(f"   v{current.version} movido para Archived.")
    else:
        print(f"❌ Candidato v{candidate.version} REPROVADO no gate.")
        print(f"   Production (v{current.version}) mantido.")
        sys.exit(2)


if __name__ == "__main__":
    main()