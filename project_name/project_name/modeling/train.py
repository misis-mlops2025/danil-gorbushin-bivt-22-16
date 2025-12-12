from __future__ import annotations

import json
from pathlib import Path

import joblib
from loguru import logger
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import typer

from project_name.config import CONFIG_DIR, MODELS_DIR, PROCESSED_DATA_DIR, load_config

app = typer.Typer(add_completion=False)


def make_model(model_type: str, random_state: int, params: dict):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier

    if model_type == "logreg":
        return LogisticRegression(random_state=random_state, **params)
    if model_type == "rf":
        return RandomForestClassifier(random_state=random_state, **params)
    if model_type == "tree":
        return DecisionTreeClassifier(random_state=random_state, **params)
    raise ValueError(f"Unknown model type: {model_type}")


def _metrics(y_true, y_prob, y_pred):
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
    }
    if isinstance(y_prob, np.ndarray) and y_prob.ndim == 1:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    else:
        out["roc_auc"] = float("nan")
    return out


@app.command()
def main(
    config: Path = typer.Option(CONFIG_DIR / "config.yaml", "--config"),
    features_path: Path = typer.Option(PROCESSED_DATA_DIR / "features.csv", "--features-path"),
    labels_path: Path = typer.Option(PROCESSED_DATA_DIR / "labels.csv", "--labels-path"),
    model_path: Path = typer.Option(MODELS_DIR / "model.pkl", "--model-path"),
    metrics_path: Path = typer.Option(MODELS_DIR / "metrics.json", "--metrics-path"),
):
    cfg = load_config(config)

    X = pd.read_csv(features_path)
    y = pd.read_csv(labels_path).iloc[:, 0].values

    model = make_model(
        cfg.model_params.model_type, cfg.model_params.random_state, cfg.selected_hparams()
    )
    model.fit(X, y)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        prob1 = proba[:, 1] if proba.ndim == 2 else proba
    else:
        prob1 = None

    y_pred = model.predict(X)
    metrics = _metrics(y, prob1 if prob1 is not None else y_pred, y_pred)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    logger.success(f"Saved model: {model_path}")
    logger.success(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    app()
