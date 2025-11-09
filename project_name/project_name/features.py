from __future__ import annotations

from pathlib import Path

from loguru import logger
import pandas as pd
import typer

from project_name.config import CONFIG_DIR, PROCESSED_DATA_DIR, load_config

app = typer.Typer(add_completion=False)


@app.command()
def main(
    config: Path = typer.Option(CONFIG_DIR / "config.yaml", "--config"),
    input_path: Path = typer.Option(PROCESSED_DATA_DIR / "dataset.csv", "--input-path"),
    features_path: Path = typer.Option(PROCESSED_DATA_DIR / "features.csv", "--features-path"),
    labels_path: Path = typer.Option(PROCESSED_DATA_DIR / "labels.csv", "--labels-path"),
):
    cfg = load_config(config)
    target = cfg.features.target_col

    df = pd.read_csv(input_path)
    if target not in df.columns:
        raise SystemExit(f"target_col '{target}' not in columns: {list(df.columns)}")

    X = df.drop(columns=[target])
    y = df[target]

    features_path.parent.mkdir(parents=True, exist_ok=True)
    X.to_csv(features_path, index=False)
    y.to_csv(labels_path, index=False, header=True)

    logger.success(f"Saved features: {features_path}  labels: {labels_path}")


if __name__ == "__main__":
    app()
