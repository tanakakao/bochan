"""DataFrame builders for visualization inputs and predictions."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from ..input_perturbation import prediction_mean_std
from ..utils import (
    candidate_result_from,
    ensure_2d,
    get_train_X,
    get_train_Y,
    infer_feature_cols,
    infer_target_cols,
    to_numpy,
)


def prediction_dataframe(
    obj: Any,
    X: Any,
    *,
    target_cols: Sequence[str] | None = None,
    uncertainty_kind: str = "epistemic",
    num_uncertainty_samples: int = 256,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """予測平均と標準偏差を DataFrame として返す。"""

    mean, std = prediction_mean_std(
        obj,
        X,
        uncertainty_kind=uncertainty_kind,
        num_uncertainty_samples=num_uncertainty_samples,
    )
    columns = infer_target_cols(obj, target_cols, mean.shape[1])
    return pd.DataFrame(mean, columns=columns), pd.DataFrame(std, columns=columns)


def training_dataframe(
    obj: Any,
    *,
    feature_cols: Sequence[str] | None = None,
    target_cols: Sequence[str] | None = None,
    X: Any | None = None,
    y: Any | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """学習 X/Y を DataFrame として返す。"""

    X_array = ensure_2d(get_train_X(obj) if X is None else X)
    y_array = ensure_2d(get_train_Y(obj) if y is None else y)
    x_columns = infer_feature_cols(obj, feature_cols, X_array.shape[1])
    y_columns = infer_target_cols(obj, target_cols, y_array.shape[1])
    return (
        pd.DataFrame(X_array, columns=x_columns),
        pd.DataFrame(y_array, columns=y_columns),
    )


def candidates_dataframe(
    obj: Any,
    *,
    candidate_result: Any | None = None,
    feature_cols: Sequence[str] | None = None,
    target_cols: Sequence[str] | None = None,
    include_prediction: bool = True,
) -> pd.DataFrame | None:
    """候補点と予測値・獲得関数値を DataFrame として返す。"""

    result = candidate_result or candidate_result_from(obj)
    if result is None or getattr(result, "candidates", None) is None:
        return None

    candidates = getattr(result, "candidates")
    X_array = ensure_2d(candidates)
    x_columns = infer_feature_cols(obj, feature_cols, X_array.shape[1])
    frame = pd.DataFrame(X_array, columns=x_columns)

    if include_prediction:
        mean_frame, std_frame = prediction_dataframe(
            obj,
            candidates,
            target_cols=target_cols,
        )
        for column in mean_frame.columns:
            frame[f"{column}_mean"] = mean_frame[column].to_numpy()
            frame[f"{column}_std"] = std_frame[column].to_numpy()

    acq_value = getattr(result, "acq_value", None)
    if acq_value is not None:
        values = np.ravel(to_numpy(acq_value))
        if len(values) == len(frame):
            frame["acq_value"] = values
        elif len(values) == 1:
            frame["acq_value"] = float(values[0])
    return frame


def get_yyplot_data(
    obj: Any,
    *,
    target_cols: Sequence[str] | None = None,
    candidate_result: Any | None = None,
):
    """YY plot 用の CV・予測・候補点データを返す。"""

    train_X = get_train_X(obj)
    prediction = prediction_dataframe(obj, train_X, target_cols=target_cols)
    candidates = candidates_dataframe(
        obj,
        candidate_result=candidate_result,
        target_cols=target_cols,
    )
    return getattr(obj, "cv_results", None), prediction, candidates


__all__ = [
    "candidates_dataframe",
    "get_yyplot_data",
    "prediction_dataframe",
    "training_dataframe",
]
