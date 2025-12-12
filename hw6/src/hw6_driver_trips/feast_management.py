from __future__ import annotations

import os
import subprocess
from pathlib import Path

from feast import FeatureStore


def apply_feast_feature_repository(feature_repo_path: Path) -> None:
    if not feature_repo_path.exists():
        raise FileNotFoundError(f"Feature repo directory not found: {feature_repo_path}")

    process = subprocess.run(
        ["feast", "apply"],
        cwd=str(feature_repo_path),
        check=False,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )

    if process.returncode != 0:
        raise RuntimeError(
            "feast apply failed\n"
            f"stdout:\n{process.stdout}\n"
            f"stderr:\n{process.stderr}\n"
        )


def create_feature_store(feature_repo_path: Path) -> FeatureStore:
    return FeatureStore(repo_path=str(feature_repo_path))
