from __future__ import annotations

from dataclasses import dataclass

from mlflow.tracking import MlflowClient


@dataclass(frozen=True)
class ModelLocator:
    run_id: str
    model_uri: str


def ensure_experiment_exists(experiment_name: str) -> str:
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is not None:
        return experiment.experiment_id
    return client.create_experiment(experiment_name)


def find_latest_training_run(experiment_name: str, model_family_tag: str) -> ModelLocator:
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise RuntimeError(f"MLflow experiment not found: {experiment_name}")

    filter_string = (
        "tags.run_type = 'training' and "
        f"tags.model_family = '{model_family_tag}' and "
        "attributes.status = 'FINISHED'"
    )

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=filter_string,
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )

    if not runs:
        raise RuntimeError("No training runs found")

    run_id = runs[0].info.run_id
    model_uri = f"runs:/{run_id}/driver_trips_regressor"
    return ModelLocator(run_id=run_id, model_uri=model_uri)
