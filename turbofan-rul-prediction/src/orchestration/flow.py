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
def train_model(model_name: str = "random_forest"):
    if model_name == "random_forest":
        subprocess.run([sys.executable, "-m", "src.training.train"], check=True)
    else:
        subprocess.run(
            [sys.executable, "-m", "src.training.train", f"model={model_name}"],
            check=True,
        )


@task
def run_predict():
    subprocess.run([sys.executable, "-m", "src.inference.predict"], check=True)


@flow
def turbofan_pipeline(model_name: str = "random_forest"):
    prepare_data()
    build_features()
    train_model(model_name=model_name)
    run_predict()


if __name__ == "__main__":
    turbofan_pipeline()