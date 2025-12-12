from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@dataclass(frozen=True)
class RegressionMetrics:
    mae: float
    rmse: float
    r2: float

    def as_dict(self) -> dict[str, float]:
        return {"mae": float(self.mae), "rmse": float(self.rmse), "r2": float(self.r2)}


def calculate_regression_metrics(y_true, y_pred):
    mean_squared_error_value = mean_squared_error(y_true, y_pred)
    root_mean_squared_error_value = float(np.sqrt(mean_squared_error_value))

    return RegressionMetrics(
        mae=float(mean_absolute_error(y_true, y_pred)),
        rmse=root_mean_squared_error_value,
        r2=float(r2_score(y_true, y_pred)),
    )
