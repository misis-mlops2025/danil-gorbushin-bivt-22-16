from __future__ import annotations

from datetime import date, timedelta

import pendulum
from airflow.decorators import dag, task
from airflow.operators.python import get_current_context

from hw6_driver_trips.data_generation import (
    DriverDatasetSpec,
    build_driver_id_list,
    ensure_driver_daily_metrics_dataset,
)
from hw6_driver_trips.feast_management import apply_feast_feature_repository, create_feature_store
from hw6_driver_trips.paths import driver_daily_metrics_parquet_path, feature_repo_dir
from hw6_driver_trips.training_pipeline import (
    TrainingConfig,
    build_entity_dataframe_for_last_n_days,
    fetch_training_data_from_feast,
    split_train_validation_by_time,
    train_regressor,
    evaluate_regressor,
    log_training_to_mlflow,
)


@dag(
    dag_id="hw6_train_driver_trips_model",
    schedule="0 3 * * *",
    start_date=pendulum.datetime(2025, 1, 1, tz="UTC"),
    catchup=False,
    tags=["hw6", "training", "feast", "mlflow"],
)
def hw6_train_driver_trips_model():

    @task
    def ensure_offline_dataset() -> str:
        context = get_current_context()
        logical_date = context["logical_date"]
        end_date_utc: date = logical_date.date()

        training_history_days = 120
        required_start = end_date_utc - timedelta(days=training_history_days - 1)

        parquet_path = driver_daily_metrics_parquet_path()
        dataset_spec = DriverDatasetSpec(number_of_drivers=100, random_seed=42)

        ensure_driver_daily_metrics_dataset(
            output_path=parquet_path,
            spec=dataset_spec,
            required_start_date_utc=required_start,
            required_end_date_utc=end_date_utc,
        )
        return str(parquet_path)

    @task
    def apply_feast_repo() -> None:
        apply_feast_feature_repository(feature_repo_path=feature_repo_dir())

    @task
    def train_model() -> str:
        context = get_current_context()
        logical_date = context["logical_date"]
        end_date_utc: date = logical_date.date()

        config = TrainingConfig()

        driver_ids = build_driver_id_list(number_of_drivers=100)
        store = create_feature_store(feature_repo_path=feature_repo_dir())

        entity_df = build_entity_dataframe_for_last_n_days(
            driver_ids=driver_ids,
            end_date_utc=end_date_utc,
            days=config.number_of_training_timestamps,
        )
        training_df = fetch_training_data_from_feast(store=store, entity_df=entity_df)

        feature_columns = ["conv_rate", "acc_rate"]
        label_column = "avg_daily_trips"

        train_df, val_df = split_train_validation_by_time(
            full_df=training_df,
            validation_timestamps=config.validation_timestamps,
        )

        model = train_regressor(
            train_df=train_df,
            feature_columns=feature_columns,
            label_column=label_column,
            config=config,
        )

        metrics = evaluate_regressor(
            model=model,
            validation_df=val_df,
            feature_columns=feature_columns,
            label_column=label_column,
        )

        run_id = log_training_to_mlflow(
            model=model,
            train_df=train_df,
            validation_df=val_df,
            feature_columns=feature_columns,
            label_column=label_column,
            metrics=metrics,
            config=config,
        )
        return run_id

    dataset_ready = ensure_offline_dataset()
    feast_ready = apply_feast_repo()
    training_run_id = train_model()

    dataset_ready >> feast_ready >> training_run_id


hw6_train_driver_trips_model()
