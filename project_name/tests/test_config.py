from project_name.config import load_config, Config, DatasetParams


def test_load_config_parses_defaults():
    cfg: Config = load_config("configs/config.yaml")
    assert cfg.model_params.model_type in {"logreg", "rf", "tree"}
    assert 0.05 <= cfg.dataset.test_size <= 0.5


def test_dataset_validator_bounds():
    ds = DatasetParams()
    assert 0.05 <= ds.test_size <= 0.5
