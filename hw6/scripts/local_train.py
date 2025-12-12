from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from hw6_driver_trips.data_generation import DriverDatasetSpec, ensure_driver_daily_metrics_dataset, build_driver_id_list
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


def main() -> None:
    end_date = date.today()
    start_date = end_date - timedelta(days=119)

    dataset_spec = DriverDatasetSpec(number_of_drivers=100, random_seed=42)
    ensure_driver_daily_metrics_dataset(
        output_path=driver_daily_metrics_parquet_path(),
        spec=dataset_spec,
        required_start_date_utc=start_date,
        required_end_date_utc=end_date,
    )

    apply_feast_feature_repository(feature_repo_path=feature_repo_dir())
    store = create_feature_store(feature_repo_path=feature_repo_dir())

    config = TrainingConfig()
    driver_ids = build_driver_id_list(number_of_drivers=100)

    entity_df = build_entity_dataframe_for_last_n_days(
        driver_ids=driver_ids,
        end_date_utc=end_date,
        days=config.number_of_training_timestamps,
    )
    full_df = fetch_training_data_from_feast(store=store, entity_df=entity_df)

    feature_columns = ["conv_rate", "acc_rate"]
    label_column = "avg_daily_trips"

    train_df, val_df = split_train_validation_by_time(full_df=full_df, validation_timestamps=config.validation_timestamps)
    model = train_regressor(train_df=train_df, feature_columns=feature_columns, label_column=label_column, config=config)
    metrics = evaluate_regressor(model=model, validation_df=val_df, feature_columns=feature_columns, label_column=label_column)

    run_id = log_training_to_mlflow(
        model=model,
        train_df=train_df,
        validation_df=val_df,
        feature_columns=feature_columns,
        label_column=label_column,
        metrics=metrics,
        config=config,
    )

    print(f"Training done. run_id={run_id}")


if __name__ == "__main__":
    main()
