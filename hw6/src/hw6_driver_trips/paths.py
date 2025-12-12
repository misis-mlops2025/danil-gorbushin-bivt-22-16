from __future__ import annotations

from pathlib import Path


def hw6_root_dir() -> Path:
    return Path(__file__).resolve().parents[2]


def feature_repo_dir() -> Path:
    return hw6_root_dir() / "feature_repo"


def feast_data_dir() -> Path:
    return feature_repo_dir() / "data"


def driver_daily_metrics_parquet_path() -> Path:
    return feast_data_dir() / "driver_daily_metrics.parquet"


def predictions_output_dir() -> Path:
    return hw6_root_dir() / "output" / "predictions"
