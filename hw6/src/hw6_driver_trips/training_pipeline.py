from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from feast import FeatureStore
from mlflow.models.signature import infer_signature
from sklearn.ensemble import RandomForestRegressor

from hw6_driver_trips.metrics import RegressionMetrics, calculate_regression_metrics


@dataclass(frozen=True)
class TrainingConfig:
    experiment_name: str = "hw6_driver_trips"
    model_family_tag: str = "driver_trips_rf_v1"
    model_artifact_path: str = "driver_trips_regressor"

    number_of_training_timestamps: int = 10
    validation_timestamps: int = 2

    random_seed: int = 7
    n_estimators: int = 250
    max_depth: int | None = 10


def _utc_midnight(day: date) -> datetime:
    return datetime(day.year, day.month, day.day, tzinfo=timezone.utc)


def build_entity_dataframe_for_last_n_days(
    driver_ids: list[int],
    end_date_utc: date,
    days: int,
) -> pd.DataFrame:
    if days <= 0:
        raise ValueError("days must be positive")

    timestamps = [_utc_midnight(end_date_utc - timedelta(days=offset)) for offset in range(days)]
    timestamps_sorted = sorted(timestamps)

    rows = [{"driver_id": d, "event_timestamp": ts} for ts in timestamps_sorted for d in driver_ids]
    return pd.DataFrame(rows)


def fetch_training_data_from_feast(store: FeatureStore, entity_df: pd.DataFrame) -> pd.DataFrame:
    feature_refs = [
        "driver_daily_metrics:conv_rate",
        "driver_daily_metrics:acc_rate",
        "driver_daily_metrics:avg_daily_trips",
    ]
    return store.get_historical_features(entity_df=entity_df, features=feature_refs).to_df()


def split_train_validation_by_time(full_df: pd.DataFrame, validation_timestamps: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if validation_timestamps <= 0:
        raise ValueError("validation_timestamps must be positive")

    unique_ts = sorted(pd.to_datetime(full_df["event_timestamp"]).unique())
    if len(unique_ts) <= validation_timestamps:
        raise ValueError("Not enough timestamps for train/validation split")

    validation_cut = set(unique_ts[-validation_timestamps:])
    is_val = pd.to_datetime(full_df["event_timestamp"]).isin(validation_cut)

    train_df = full_df.loc[~is_val].copy()
    val_df = full_df.loc[is_val].copy()
    return train_df, val_df


def train_regressor(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    label_column: str,
    config: TrainingConfig,
) -> RandomForestRegressor:
    X = train_df[feature_columns]
    y = train_df[label_column].astype("float32")

    model = RandomForestRegressor(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        random_state=config.random_seed,
        n_jobs=-1,
    )
    model.fit(X, y)
    return model


def evaluate_regressor(
    model: RandomForestRegressor,
    validation_df: pd.DataFrame,
    feature_columns: list[str],
    label_column: str,
) -> RegressionMetrics:
    X_val = validation_df[feature_columns]
    y_true = validation_df[label_column].astype("float32").to_numpy()
    y_pred = model.predict(X_val)
    return calculate_regression_metrics(y_true=y_true, y_pred=y_pred)


def log_training_to_mlflow(
    model: RandomForestRegressor,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    feature_columns: list[str],
    label_column: str,
    metrics: RegressionMetrics,
    config: TrainingConfig,
) -> str:
    mlflow.set_experiment(config.experiment_name)

    X_example = train_df[feature_columns].head(20)
    y_example = model.predict(X_example)
    signature = infer_signature(X_example, y_example)

    with mlflow.start_run(run_name="training_driver_trips_model") as run:
        mlflow.set_tag("run_type", "training")
        mlflow.set_tag("model_family", config.model_family_tag)

        mlflow.log_param("feature_columns", ",".join(feature_columns))
        mlflow.log_param("label_column", label_column)
        mlflow.log_param("n_train_rows", int(train_df.shape[0]))
        mlflow.log_param("n_validation_rows", int(validation_df.shape[0]))

        mlflow.log_param("n_estimators", config.n_estimators)
        mlflow.log_param("max_depth", -1 if config.max_depth is None else int(config.max_depth))

        mlflow.log_metrics(
            {
                "val_mae": metrics.mae,
                "val_rmse": metrics.rmse,
                "val_r2": metrics.r2,
            }
        )

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=config.model_artifact_path,
            signature=signature,
            input_example=X_example,
        )

        debug_sample = pd.concat([train_df.head(30), validation_df.head(30)], ignore_index=True)
        sample_path = Path("/tmp/hw6_training_sample.parquet")
        debug_sample.to_parquet(sample_path, index=False)
        mlflow.log_artifact(str(sample_path), artifact_path="debug_samples")

        return run.info.run_id
