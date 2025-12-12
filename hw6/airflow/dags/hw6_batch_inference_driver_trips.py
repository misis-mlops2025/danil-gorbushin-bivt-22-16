from __future__ import annotations

from datetime import date, datetime

import mlflow
import pendulum
from airflow.decorators import dag, task
from airflow.operators.python import get_current_context

from hw6_driver_trips.data_generation import (
    DriverDatasetSpec,
    build_driver_id_list,
    ensure_driver_daily_metrics_dataset,
)
from hw6_driver_trips.feast_management import apply_feast_feature_repository, create_feature_store
from hw6_driver_trips.inference_pipeline import (
    InferenceConfig,
    build_entity_dataframe_for_inference_date,
    fetch_inference_features_from_feast,
    run_batch_inference,
    save_predictions_parquet,
    log_inference_to_mlflow,
)
from hw6_driver_trips.mlflow_io import find_latest_training_run
from hw6_driver_trips.paths import driver_daily_metrics_parquet_path, feature_repo_dir, predictions_output_dir


def _parse_iso_date(date_text: str) -> date:
    return datetime.strptime(date_text, "%Y-%m-%d").date()


@dag(
    dag_id="hw6_batch_inference_driver_trips",
    schedule=None,
    start_date=pendulum.datetime(2025, 1, 1, tz="UTC"),
    catchup=False,
    tags=["hw6", "inference", "feast", "mlflow"],
)
def hw6_batch_inference_driver_trips():

    @task
    def apply_feast_repo() -> None:
        apply_feast_feature_repository(feature_repo_path=feature_repo_dir())

    @task
    def run_inference() -> str:
        context = get_current_context()

        dag_run = context.get("dag_run")
        dag_conf = dag_run.conf if dag_run and dag_run.conf else {}

        logical_date = context["logical_date"]
        default_date = logical_date.date()

        inference_date_text = dag_conf.get("inference_date")
        inference_date_utc = _parse_iso_date(inference_date_text) if inference_date_text else default_date

        dataset_spec = DriverDatasetSpec(number_of_drivers=100, random_seed=42)
        ensure_driver_daily_metrics_dataset(
            output_path=driver_daily_metrics_parquet_path(),
            spec=dataset_spec,
            required_start_date_utc=inference_date_utc,
            required_end_date_utc=inference_date_utc,
        )

        training_locator = find_latest_training_run(
            experiment_name="hw6_driver_trips",
            model_family_tag="driver_trips_rf_v1",
        )
        model = mlflow.pyfunc.load_model(training_locator.model_uri)

        driver_ids = build_driver_id_list(number_of_drivers=100)
        store = create_feature_store(feature_repo_path=feature_repo_dir())

        entity_df = build_entity_dataframe_for_inference_date(
            driver_ids=driver_ids,
            inference_date_utc=inference_date_utc,
        )
        features_df = fetch_inference_features_from_feast(store=store, entity_df=entity_df)

        feature_columns = ["conv_rate", "acc_rate"]
        predictions_df = run_batch_inference(model=model, features_df=features_df, feature_columns=feature_columns)

        output_path = predictions_output_dir() / f"predictions_{inference_date_utc.isoformat()}.parquet"
        saved_path = save_predictions_parquet(predictions_df=predictions_df, output_path=output_path)

        run_id = log_inference_to_mlflow(
            inference_date_utc=inference_date_utc,
            model_locator=training_locator,
            predictions_parquet_path=saved_path,
            features_df_with_label=features_df,
            feature_columns=feature_columns,
            config=InferenceConfig(),
        )
        return run_id

    feast_ready = apply_feast_repo()
    inference_run = run_inference()

    feast_ready >> inference_run


hw6_batch_inference_driver_trips()
