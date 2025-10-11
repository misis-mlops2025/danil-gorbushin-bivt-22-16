from __future__ import annotations
from sklearn.datasets import make_classification
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger
from tqdm import tqdm
import typer

from project_name.config import (
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    CONFIG_DIR,
    load_config,
)

app = typer.Typer(add_completion=False)


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
        .str.replace(r"[^0-9a-zA-Z_]", "", regex=True)
    )
    return df


def _drop_constant_columns(df: pd.DataFrame) -> pd.DataFrame:
    nunique = df.nunique(dropna=False)
    keep = nunique[nunique > 1].index
    dropped = [c for c in df.columns if c not in keep]
    if dropped:
        logger.info(f"Dropping constant columns: {dropped}")
    return df[keep]


def _impute_numeric_median(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.select_dtypes(include="number").columns:
        if df[col].isna().any():
            med = df[col].median()
            df[col] = df[col].fillna(med)
            logger.info(f"Impute NaN in '{col}' with median={med:.4g}")
    return df


def _clip_percentiles(df: pd.DataFrame, p_lo: float, p_hi: float) -> pd.DataFrame:
    df = df.copy()
    for col in df.select_dtypes(include="number").columns:
        lo, hi = df[col].quantile(p_lo / 100.0), df[col].quantile(p_hi / 100.0)
        df[col] = df[col].clip(lower=lo, upper=hi)
    logger.info(f"Clipped to [{p_lo}%, {p_hi}%]")
    return df


def _apply_binary_map(
        df: pd.DataFrame,
        mapping: dict[str, dict[str, list[Any]]],
) -> pd.DataFrame:
    df = df.copy()
    precomputed = {
        col: (
            {str(v).lower() for v in spec.get("true", [])},
            {str(v).lower() for v in spec.get("false", [])},
        )
        for col, spec in mapping.items()
    }
    for col, (true_vals, false_vals) in precomputed.items():
        if col not in df.columns:
            logger.warning("binary_map: column '%s' not found, skipping", col)
            continue

        def _to01(x: Any) -> Any:
            s = str(x).lower()
            if s in true_vals:
                return 1
            if s in false_vals:
                return 0
            return x

        df[col] = df[col].map(_to01)
        logger.info("binary_map applied to '%s'", col)
    return df


@app.command()
def main(
        config: Path = typer.Option(CONFIG_DIR / "config.yaml", "--config"),
        input_path: Path = typer.Option(RAW_DATA_DIR / "dataset.csv", "--input-path", "-i"),
        output_path: Path = typer.Option(PROCESSED_DATA_DIR / "dataset.csv", "--output-path", "-o"),
):
    cfg = load_config(config)

    if not input_path.exists():
        logger.warning(f"{input_path} not found. Generating synthetic dataset via make_classification()")
        input_path.parent.mkdir(parents=True, exist_ok=True)
        X, y = make_classification(
            n_samples=cfg.dataset.n_samples,
            n_features=cfg.dataset.n_features,
            n_informative=cfg.dataset.n_informative,
            n_redundant=cfg.dataset.n_redundant,
            class_sep=cfg.dataset.class_sep,
            random_state=cfg.dataset.random_state,
        )
        df_gen = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        df_gen["target"] = y
        df_gen.to_csv(input_path, index=False)
        logger.success(f"Generated and saved raw dataset to {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Reading: {input_path}")
    df = pd.read_csv(input_path)

    steps = [("standardize_columns", _standardize_columns)]
    if cfg.preprocess.drop_duplicates:
        steps.append(("drop_duplicates", lambda d: d.drop_duplicates()))
    if cfg.preprocess.drop_constant:
        steps.append(("drop_constant", _drop_constant_columns))
    if cfg.preprocess.impute_numeric:
        steps.append(("impute_numeric_median", _impute_numeric_median))
    if isinstance(cfg.preprocess.clip_percentiles, list) and len(cfg.preprocess.clip_percentiles) == 2:
        lo, hi = float(cfg.preprocess.clip_percentiles[0]), float(cfg.preprocess.clip_percentiles[1])
        steps.append(("clip_percentiles", lambda d, _lo=lo, _hi=hi: _clip_percentiles(d, _lo, _hi)))
    if cfg.preprocess.binary_map:
        steps.append(("binary_map", lambda d, m=cfg.preprocess.binary_map: _apply_binary_map(d, m)))

    logger.info("Processing dataset...")
    for name, fn in tqdm(steps, total=len(steps)):
        df = fn(df)
        logger.info(f"done: {name} (shape={df.shape})")

    df.to_csv(output_path, index=False)
    logger.success(f"Saved: {output_path} (shape={df.shape})")


if __name__ == "__main__":
    app()
