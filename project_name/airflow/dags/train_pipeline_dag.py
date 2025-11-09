from __future__ import annotations
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

from project_name.dataset import main as process_cli
from project_name.features import main as features_cli
from project_name.modeling.train import main as train_cli

CONFIG_PATH = Path("/opt/airflow/configs/config.yaml")
DATA_PROCESSED = Path("/opt/airflow/data")
MODELS_DIR = Path("/opt/airflow/project/models")

default_args = {
    "owner": "mlops",
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

with DAG(
        dag_id="train_pipeline",
        description="HW4: training pipeline (generate -> preprocess -> train)",
        start_date=datetime(2024, 1, 1),
        schedule_interval=None,
        catchup=False,
        default_args=default_args,
        tags=["hw4", "train"],
) as dag:
    def step_process():
        DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
        process_cli(
            config=CONFIG_PATH,
            input_path=Path("/opt/airflow/data/raw/dataset.csv"),
            output_path=DATA_PROCESSED / "dataset.csv",
        )


    def step_features():
        features_cli(
            config=CONFIG_PATH,
            input_path=DATA_PROCESSED / "dataset.csv",
            features_path=DATA_PROCESSED / "features.csv",
            labels_path=DATA_PROCESSED / "labels.csv",
        )


    def step_train():
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        train_cli(
            config=CONFIG_PATH,
            features_path=DATA_PROCESSED / "features.csv",
            labels_path=DATA_PROCESSED / "labels.csv",
            model_path=MODELS_DIR / "model.pkl",
            metrics_path=MODELS_DIR / "metrics.json",
        )


    t1 = PythonOperator(task_id="process", python_callable=step_process)
    t2 = PythonOperator(task_id="features", python_callable=step_features)
    t3 = PythonOperator(task_id="train", python_callable=step_train)

    t1 >> t2 >> t3
