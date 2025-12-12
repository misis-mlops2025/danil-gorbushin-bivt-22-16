import pandas as pd
from pathlib import Path
from project_name.modeling.train import main as train_cli


def test_train_and_metrics(tmp_path: Path):
    proc = tmp_path / "processed";
    proc.mkdir()
    X = pd.DataFrame({"x1": [0, 1, 1, 0, 1], "x2": [1, 0, 1, 0, 1]})
    y = pd.DataFrame({"target": [0, 1, 1, 0, 1]})
    (proc / "features.csv").write_text(X.to_csv(index=False), encoding="utf-8")
    (proc / "labels.csv").write_text(y.to_csv(index=False), encoding="utf-8")

    models_dir = tmp_path / "models";
    models_dir.mkdir()

    train_cli(
        features_path=proc / "features.csv",
        labels_path=proc / "labels.csv",
        model_path=models_dir / "model.pkl",
        metrics_path=models_dir / "metrics.json",
        config=Path("configs/config.yaml"),
    )
    assert (models_dir / "model.pkl").exists()
    assert (models_dir / "metrics.json").exists()
