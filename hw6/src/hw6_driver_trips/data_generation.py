from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DriverDatasetSpec:
    number_of_drivers: int = 100
    random_seed: int = 42


_EPOCH = date(2020, 1, 1)


def _utc_midnight(day: date) -> datetime:
    return datetime(day.year, day.month, day.day, tzinfo=timezone.utc)


def build_driver_id_list(number_of_drivers: int) -> list[int]:
    if number_of_drivers <= 0:
        raise ValueError("number_of_drivers must be positive")
    return list(range(1, number_of_drivers + 1))


def _read_existing_date_range(parquet_path: Path) -> tuple[date, date]:
    existing = pd.read_parquet(parquet_path, columns=["event_timestamp"])
    timestamps = pd.to_datetime(existing["event_timestamp"], utc=True)
    return timestamps.min().date(), timestamps.max().date()


def generate_driver_daily_metrics_parquet(
    output_path: Path,
    spec: DriverDatasetSpec,
    start_date_utc: date,
    end_date_utc: date,
) -> Path:
    if start_date_utc > end_date_utc:
        raise ValueError("start_date_utc must be <= end_date_utc")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    driver_rng = np.random.default_rng(spec.random_seed)
    driver_ids = np.asarray(build_driver_id_list(spec.number_of_drivers), dtype=np.int64)

    base_conv_rate = driver_rng.uniform(0.05, 0.35, size=spec.number_of_drivers)
    base_acc_rate = driver_rng.uniform(0.60, 0.98, size=spec.number_of_drivers)
    driver_skill = driver_rng.normal(0.0, 1.0, size=spec.number_of_drivers)

    day_count = (end_date_utc - start_date_utc).days + 1
    frames: list[pd.DataFrame] = []

    for offset in range(day_count):
        current_day = start_date_utc + timedelta(days=offset)
        global_day_index = (current_day - _EPOCH).days

        day_rng = np.random.default_rng(spec.random_seed + global_day_index)

        weekly_wave = 0.04 * np.sin(2 * np.pi * (global_day_index / 7.0))
        monthly_wave = 0.02 * np.cos(2 * np.pi * (global_day_index / 30.0))

        conv_rate = base_conv_rate + weekly_wave + day_rng.normal(0.0, 0.02, size=spec.number_of_drivers)
        acc_rate = base_acc_rate + monthly_wave + day_rng.normal(0.0, 0.015, size=spec.number_of_drivers)

        conv_rate = np.clip(conv_rate, 0.01, 0.99)
        acc_rate = np.clip(acc_rate, 0.01, 0.99)

        avg_daily_trips = (
            20.0
            + 380.0 * conv_rate
            + 140.0 * acc_rate
            + 8.0 * driver_skill
            + 15.0 * np.maximum(conv_rate - 0.2, 0)
            + day_rng.normal(0.0, 10.0, size=spec.number_of_drivers)
        )
        avg_daily_trips = np.clip(avg_daily_trips, 1.0, None)

        frames.append(
            pd.DataFrame(
                {
                    "driver_id": driver_ids,
                    "event_timestamp": np.repeat(pd.Timestamp(_utc_midnight(current_day)), spec.number_of_drivers),
                    "conv_rate": conv_rate.astype("float32"),
                    "acc_rate": acc_rate.astype("float32"),
                    "avg_daily_trips": avg_daily_trips.astype("float32"),
                }
            )
        )

    full_df = pd.concat(frames, ignore_index=True)
    full_df.sort_values(["event_timestamp", "driver_id"], inplace=True)
    full_df.to_parquet(output_path, index=False)
    return output_path


def ensure_driver_daily_metrics_dataset(
    output_path: Path,
    spec: DriverDatasetSpec,
    required_start_date_utc: date,
    required_end_date_utc: date,
    extra_days_before: int = 0,
    extra_days_after: int = 0,
) -> Path:
    if required_start_date_utc > required_end_date_utc:
        raise ValueError("required_start_date_utc must be <= required_end_date_utc")

    target_start = required_start_date_utc - timedelta(days=extra_days_before)
    target_end = required_end_date_utc + timedelta(days=extra_days_after)

    if output_path.exists():
        existing_start, existing_end = _read_existing_date_range(output_path)
        if existing_start <= target_start and existing_end >= target_end:
            return output_path

        target_start = min(target_start, existing_start)
        target_end = max(target_end, existing_end)

    return generate_driver_daily_metrics_parquet(
        output_path=output_path,
        spec=spec,
        start_date_utc=target_start,
        end_date_utc=target_end,
    )
