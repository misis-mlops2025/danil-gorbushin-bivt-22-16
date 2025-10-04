from pathlib import Path
import pandas as pd
from project_name.modeling.train import main as train_cli
from project_name.modeling.predict import main as predict_cli


def test_train_and_predict(tmp_path: Path):
    proc = tmp_path / "processed";
    proc.mkdir()
    models = tmp_path / "models";
    models.mkdir()

    X = pd.DataFrame({"x1": [0, 1, 1, 0, 1], "x2": [1, 0, 1, 0, 1]})
    y = pd.DataFrame({"target": [0, 1, 1, 0, 1]})
    (proc / "features.csv").write_text(X.to_csv(index=False), encoding="utf-8")
    (proc / "labels.csv").write_text(y.to_csv(index=False), encoding="utf-8")

    train_cli(
        features_path=proc / "features.csv",
        labels_path=proc / "labels.csv",
        model_path=models / "model.pkl",
        metrics_path=models / "metrics.json",
        config=Path("configs/config.yaml"),
    )
    assert (models / "model.pkl").exists()
    assert (models / "metrics.json").exists()

    Xtest = pd.DataFrame({"x1": [0, 1], "x2": [1, 0]})
    (proc / "test_features.csv").write_text(Xtest.to_csv(index=False), encoding="utf-8")
    predict_cli(
        features_path=proc / "test_features.csv",
        model_path=models / "model.pkl",
        predictions_path=proc / "predictions.csv",
    )
    preds = pd.read_csv(proc / "predictions.csv")
    assert len(preds) == 2
