from __future__ import annotations
import hydra
import mlflow
from mlflow.tracking import MlflowClient
from omegaconf import DictConfig

MODEL_NAME = "turbofan_rul_model"

@hydra.main(version_base = None, config_path = "../../configs", config_name = "config")
def main(cfg : DictConfig):
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    client = MlflowClient()
    
    exp = client.get_experiment_by_name(cfg.mlflow.experiment_name)
    if exp is None:
        raise RuntimeError(f"Experimento '{cfg.mlflow.experiment_name}' não existe.")
    runs = client.search_runs(
        [exp.experiment_id],
        filter_string  = "tags.mlflow.runName = 'lstm'",
        order_by = ["start_time DESC"],
        max_results = 1,
    )
    if not runs:
        raise RuntimeError("Nenhuma execução 'lstm' encontrada no experimento.")
    
    run = runs[0]
    model_uri = f"runs:/{run.info.run_id}/model"
    
    try : 
        client.create_registered_model(MODEL_NAME)
    except Exception as e:
        print(f"Modelo '{MODEL_NAME}' já existe. Continuando...")
        pass 
    
    mv = client.create_model_version(
        name = MODEL_NAME,
        source = model_uri,
        run_id = run.info.run_id,
    )
    client.set_model_version_tag(MODEL_NAME, mv.version, "stage", "Staging")
    client.set_model_version_tag(
        MODEL_NAME, mv.version, "test_rmse", run.data.metrics.get("test_rmse", "NA")
    )
    client.set_model_version_tag(
        MODEL_NAME, mv.version, "test_r2", run.data.metrics.get("test_r2", "NA")
    )
    print(f"Registrado: {MODEL_NAME} v{mv.version} -> Staging")
    print(f"Run ID: {run.info.run_id}")
    print(f"test_rmse: {run.data.metrics.get('test_rmse')}")
    print(f"test_r2:   {run.data.metrics.get('test_r2')}")


if __name__ == "__main__":
    main()
    