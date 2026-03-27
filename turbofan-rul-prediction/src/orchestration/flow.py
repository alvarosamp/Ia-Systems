from __future__ import annotations

import subprocess

from prefect import flow, task


@task(name="prepare-data")
def prepare_data():
    subprocess.run(["python", "-m", "src.data.prepare"], check=True)


@task(name="build-features")
def build_features():
    subprocess.run(["python", "-m", "src.features.build_features"], check=True)


@task(name="train-model")
def train_model(model_name: str = "xgboost"):
    if model_name == "random_forest":
        subprocess.run(["python", "-m", "src.training.train"], check=True)
    else:
        subprocess.run(
            ["python", "-m", "src.training.train", f"model={model_name}"],
            check=True,
        )


@task(name="predict")
def run_predict():
    subprocess.run(["python", "-m", "src.inference.predict"], check=True)


@task(name="run-drift-check")
def run_drift():
    subprocess.run(["python", "-m", "src.monitoring.drift"], check=True)


@flow(name="turbofan-rul-pipeline")
def turbofan_pipeline(model_name: str = "random_forest"):
    prepare_data()
    build_features()
    train_model(model_name=model_name)
    run_predict()
    run_drift()


if __name__ == "__main__":
    turbofan_pipeline()