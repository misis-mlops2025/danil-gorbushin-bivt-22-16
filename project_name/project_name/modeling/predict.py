from __future__ import annotations
from pathlib import Path

import joblib
import pandas as pd
from loguru import logger
import typer

from project_name.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer(add_completion=False)


@app.command()
def main(
        features_path: Path = typer.Option(PROCESSED_DATA_DIR / "test_features.csv", "--features-path"),
        model_path: Path = typer.Option(MODELS_DIR / "model.pkl", "--model-path"),
        predictions_path: Path = typer.Option(PROCESSED_DATA_DIR / "predictions.csv", "--predictions-path")

):
    model = joblib.load(model_path)
    X = pd.read_csv(features_path)
    preds = model.predict(X)
    pd.Series(preds, name="prediction").to_csv(predictions_path, index=False)
    logger.success(f"Saved predictions: {predictions_path}")


if __name__ == "__main__":
    app()
