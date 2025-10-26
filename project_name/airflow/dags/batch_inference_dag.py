from __future__ import annotations
from datetime import datetime, timedelta
from pathlib import Path
import os

from airflow import DAG
from airflow.operators.python import PythonOperator, PythonVirtualenvOperator
from airflow.sensors.python import PythonSensor

from project_name.modeling.predict import main as predict_cli

DATA_PROCESSED = Path("/opt/airflow/data")
INCOMING_FILE = Path("/opt/data/new_data.csv")
MODELS_DIR = Path("/opt/airflow/project/models")
PRED_OUT_DIR = DATA_PROCESSED / "predictions"

default_args = {
    "owner": "mlops",
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}


def _file_exists() -> bool:
    return INCOMING_FILE.exists()


def _run_predict_and_cleanup(**context):
    PRED_OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = context["ts_nodash"]
    out_path = PRED_OUT_DIR / f"pred_{ts}.csv"
    predict_cli(
        features_path=INCOMING_FILE,
        model_path=MODELS_DIR / "model.pkl",
        predictions_path=out_path,
    )
    os.remove(INCOMING_FILE)


with DAG(
    dag_id="batch_inference",
    description="HW4: batch inference with file sensor",
    start_date=datetime(2024, 1, 1),
    schedule_interval=timedelta(minutes=5),
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
    tags=["hw4", "inference"],
) as dag:
    wait_for_file = PythonSensor(
        task_id="wait_for_new_data_csv",
        python_callable=_file_exists,
        poke_interval=30,
        timeout=24 * 60 * 60,
        mode="reschedule",
        soft_fail=False,
    )
    run_predict = PythonOperator(
        task_id="predict_and_cleanup",
        python_callable=_run_predict_and_cleanup,
        provide_context=True,
    )

    wait_for_file >> run_predict
