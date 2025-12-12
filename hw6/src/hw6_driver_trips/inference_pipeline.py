from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import mlflow
import pandas as pd
from feast import FeatureStore

from hw6_driver_trips.metrics import calculate_regression_metrics
from hw6_driver_trips.mlflow_io import ModelLocator, ensure_experiment_exists


@dataclass(frozen=True)
class InferenceConfig:
    experiment_name: str = "hw6_driver_trips_inference"
    model_family_tag: str = "driver_trips_rf_v1"


def _utc_midnight(day: date) -> datetime:
    return datetime(day.year, day.month, day.day, tzinfo=timezone.utc)


def build_entity_dataframe_for_inference_date(driver_ids: list[int], inference_date_utc: date) -> pd.DataFrame:
    ts = _utc_midnight(inference_date_utc)
    return pd.DataFrame({"driver_id": driver_ids, "event_timestamp": [ts] * len(driver_ids)})


def fetch_inference_features_from_feast(store: FeatureStore, entity_df: pd.DataFrame) -> pd.DataFrame:
    feature_refs = [
        "driver_daily_metrics:conv_rate",
        "driver_daily_metrics:acc_rate",
        "driver_daily_metrics:avg_daily_trips",
    ]
    return store.get_historical_features(entity_df=entity_df, features=feature_refs).to_df()


def run_batch_inference(model, features_df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    preds = model.predict(features_df[feature_columns]).astype("float32")
    out = features_df[["driver_id", "event_timestamp"]].copy()
    out["predicted_avg_daily_trips"] = preds
    return out


def save_predictions_parquet(predictions_df: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_parquet(output_path, index=False)
    return output_path


def log_inference_to_mlflow(
    inference_date_utc: date,
    model_locator: ModelLocator,
    predictions_parquet_path: Path,
    features_df_with_label: pd.DataFrame,
    feature_columns: list[str],
    config: InferenceConfig,
) -> str:
    ensure_experiment_exists(config.experiment_name)
    mlflow.set_experiment(config.experiment_name)

    with mlflow.start_run(run_name=f"inference_{inference_date_utc.isoformat()}") as run:
        mlflow.set_tag("run_type", "inference")
        mlflow.set_tag("model_family", config.model_family_tag)
        mlflow.set_tag("model_source_run_id", model_locator.run_id)

        mlflow.log_param("inference_date", inference_date_utc.isoformat())
        mlflow.log_param("feature_columns", ",".join(feature_columns))

        mlflow.log_artifact(str(predictions_parquet_path), artifact_path="predictions")

        if "avg_daily_trips" in features_df_with_label.columns:
            y_true = features_df_with_label["avg_daily_trips"].astype("float32").to_numpy()
            y_pred = pd.read_parquet(predictions_parquet_path)["predicted_avg_daily_trips"].astype("float32").to_numpy()
            metrics = calculate_regression_metrics(y_true=y_true, y_pred=y_pred)
            mlflow.log_metrics({"mae": metrics.mae, "rmse": metrics.rmse, "r2": metrics.r2})

        return run.info.run_id
