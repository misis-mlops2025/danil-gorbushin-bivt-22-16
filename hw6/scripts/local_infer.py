from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import mlflow

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from hw6_driver_trips.data_generation import DriverDatasetSpec, ensure_driver_daily_metrics_dataset, build_driver_id_list
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True, help="YYYY-MM-DD")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inference_date = datetime.strptime(args.date, "%Y-%m-%d").date()

    apply_feast_feature_repository(feature_repo_path=feature_repo_dir())

    dataset_spec = DriverDatasetSpec(number_of_drivers=100, random_seed=42)
    ensure_driver_daily_metrics_dataset(
        output_path=driver_daily_metrics_parquet_path(),
        spec=dataset_spec,
        required_start_date_utc=inference_date,
        required_end_date_utc=inference_date,
    )

    training_locator = find_latest_training_run(
        experiment_name="hw6_driver_trips",
        model_family_tag="driver_trips_rf_v1",
    )
    model = mlflow.pyfunc.load_model(training_locator.model_uri)

    store = create_feature_store(feature_repo_path=feature_repo_dir())
    driver_ids = build_driver_id_list(number_of_drivers=100)

    entity_df = build_entity_dataframe_for_inference_date(driver_ids=driver_ids, inference_date_utc=inference_date)
    features_df = fetch_inference_features_from_feast(store=store, entity_df=entity_df)

    feature_columns = ["conv_rate", "acc_rate"]
    predictions_df = run_batch_inference(model=model, features_df=features_df, feature_columns=feature_columns)

    output_path = predictions_output_dir() / f"predictions_{inference_date.isoformat()}.parquet"
    saved_path = save_predictions_parquet(predictions_df=predictions_df, output_path=output_path)

    run_id = log_inference_to_mlflow(
        inference_date_utc=inference_date,
        model_locator=training_locator,
        predictions_parquet_path=saved_path,
        features_df_with_label=features_df,
        feature_columns=feature_columns,
        config=InferenceConfig(),
    )

    print(f"Saved predictions: {saved_path}")
    print(f"Inference run_id={run_id}")


if __name__ == "__main__":
    main()
