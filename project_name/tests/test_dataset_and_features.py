import pandas as pd
from pathlib import Path
from project_name.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from project_name.dataset import main as process_cli
from project_name.features import main as features_cli


def test_process_and_features(tmp_path: Path):
    raw = tmp_path / "raw";
    raw.mkdir()
    proc = tmp_path / "processed";
    proc.mkdir()

    df = pd.DataFrame({
        "A ": [1, 2, 2, None],
        "B": [10, 20, 20, 30],
        "target": [0, 1, 1, 0],
    })
    src = raw / "dataset.csv"
    df.to_csv(src, index=False)

    process_cli(
        config=Path("configs/config.yaml"),
        input_path=src,
        output_path=proc / "dataset.csv",
    )
    out = pd.read_csv(proc / "dataset.csv")
    assert "a" in out.columns and "b" in out.columns
    assert len(out) <= len(df)

    features_cli(
        config=Path("configs/config.yaml"),
        input_path=proc / "dataset.csv",
        features_path=proc / "features.csv",
        labels_path=proc / "labels.csv",
    )
    X = pd.read_csv(proc / "features.csv")
    y = pd.read_csv(proc / "labels.csv")
    assert "target" not in X.columns
    assert y.shape[1] == 1
