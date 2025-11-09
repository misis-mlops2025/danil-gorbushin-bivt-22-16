from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel, Field, field_validator
import yaml

load_dotenv()

PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
CONFIG_DIR = PROJ_ROOT / "configs"

try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass

ModelType = Literal["logreg", "rf", "tree"]


class DatasetParams(BaseModel):
    n_samples: int = 1200
    n_features: int = 12
    n_informative: int = 8
    n_redundant: int = 2
    class_sep: float = 1.4
    random_state: int = 42
    test_size: float = 0.25
    stratify: bool = True

    @field_validator("test_size")
    @classmethod
    def _check_test_size(cls, v: float) -> float:
        if not 0.05 <= v <= 0.5:
            raise ValueError("test_size должен быть в диапазоне [0.05, 0.5]")
        return v


class PreprocessParams(BaseModel):
    drop_duplicates: bool = True
    drop_constant: bool = True
    impute_numeric: bool = True
    clip_percentiles: list[float] = Field(default_factory=lambda: [1.0, 99.0])
    binary_map: dict[str, dict[str, list[Any]]] = Field(default_factory=dict)


class FeaturesParams(BaseModel):
    target_col: str = "target"


class ModelParams(BaseModel):
    model_type: ModelType = "logreg"
    random_state: int = 42
    logreg: dict[str, Any] = Field(default_factory=lambda: {"C": 1.2, "max_iter": 200})
    rf: dict[str, Any] = Field(default_factory=lambda: {"n_estimators": 160, "max_depth": 7})
    tree: dict[str, Any] = Field(default_factory=lambda: {"max_depth": 6})


class TrainParams(BaseModel):
    scoring: Literal["roc_auc", "accuracy", "f1"] = "roc_auc"
    save_dir: str = "models"


class Config(BaseModel):
    random_state: int = 42
    dataset: DatasetParams = DatasetParams()
    preprocess: PreprocessParams = PreprocessParams()
    features: FeaturesParams = FeaturesParams()
    model_params: ModelParams = ModelParams()
    train: TrainParams = TrainParams()

    def selected_hparams(self) -> dict[str, Any]:
        mt = self.model_params.model_type
        return self.model_params.model_dump()[mt]


def load_config(config_path: str | Path = CONFIG_DIR / "config.yaml") -> Config:
    path = Path(config_path)
    with open(path, "r", encoding="utf-8") as fin:
        raw = yaml.safe_load(fin) or {}
    cfg = Config(**raw)
    logger.info(f"Loaded config from: {path}")
    return cfg
