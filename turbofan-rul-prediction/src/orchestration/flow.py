from __future__ import annotations

import subprocess
import sys

from prefect import flow, task


@task
def prepare_data():
    subprocess.run([sys.executable, "-m", "src.data.prepare"], check=True)


@task
def build_features():
    subprocess.run([sys.executable, "-m", "src.features.build_features"], check=True)


@task
def train_lstm():
    subprocess.run([sys.executable, "-m", "src.dl.train_lstm"], check=True)


@task
def train_baseline(model_name: str = "xgboost"):
    """Treina baseline tabular (xgboost ou random_forest) pra comparação."""
    subprocess.run(
        [sys.executable, "-m", "src.training.train", f"model={model_name}"],
        check=True,
    )


@task
def run_predict():
    subprocess.run([sys.executable, "-m", "src.inference.predict"], check=True)


@flow(name="turbofan-rul-pipeline")
def turbofan_pipeline(
    train_baseline_model: bool = True,
    baseline_name: str = "xgboost",
):
    prepare_data()
    build_features()
    train_lstm()
    if train_baseline_model:
        train_baseline(model_name=baseline_name)
    run_predict()


if __name__ == "__main__":
    turbofan_pipeline()