"""Observation-state mixin for the canonical tabular optimizer class."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from bochan.api import ExperimentFailureConfig
from bochan.api.experiment_failure import attach_observation_state

from .config import ColumnKey, TabularDataConfig
from .observation_data import (
    ObservationTabularDataset,
    dataframe_to_observation_tensors,
    numpy_to_observation_tensors,
)


def _resolved_observation_config(
    base: TabularDataConfig,
    *,
    target_missing_strategy: str | None,
    experiment_status_col: ColumnKey | None,
) -> TabularDataConfig:
    """Apply observation-only direct fields to one tabular data config."""

    return replace(
        base,
        target_missing_strategy=(
            base.target_missing_strategy
            if target_missing_strategy is None
            else str(target_missing_strategy)
        ),
        experiment_status_col=(
            base.experiment_status_col
            if experiment_status_col is None
            else experiment_status_col
        ),
    )


def _uses_observation_conversion(config: TabularDataConfig) -> bool:
    return (
        str(config.target_missing_strategy).strip().lower() == "keep"
        or config.experiment_status_col is not None
    )


class ObservationTabularMixin:
    """Add explicit partial/failure observation semantics to a tabular optimizer.

    The mixin is intentionally not another public optimizer class. It is composed
    into the existing final composition-aware ``TabularBayesianOptimizer`` so the
    package has one canonical public class and all existing composition features
    remain on that same class object.
    """

    def __init__(
        self,
        *args: Any,
        data_config: TabularDataConfig | None = None,
        target_missing_strategy: str | None = None,
        experiment_status_col: ColumnKey | None = None,
        failure_config: ExperimentFailureConfig | None = None,
        **kwargs: Any,
    ) -> None:
        resolved = _resolved_observation_config(
            data_config or TabularDataConfig(),
            target_missing_strategy=target_missing_strategy,
            experiment_status_col=experiment_status_col,
        )
        self.failure_config = failure_config
        super().__init__(*args, data_config=resolved, **kwargs)

    def _to_dataset(
        self,
        data: Any,
        y: Any | None = None,
        *,
        data_config: TabularDataConfig | None = None,
        feature_names: Any = None,
        target_names: Any = None,
    ):
        config = data_config or self.data_config
        if not _uses_observation_conversion(config):
            return super()._to_dataset(
                data,
                y,
                data_config=config,
                feature_names=feature_names,
                target_names=target_names,
            )

        try:
            import pandas as pd
        except ImportError:
            pd = None
        if pd is not None and isinstance(data, pd.DataFrame):
            return dataframe_to_observation_tensors(data, config)
        return numpy_to_observation_tensors(
            data,
            y,
            config,
            feature_names=feature_names,
            target_names=target_names,
        )

    def fit(
        self,
        data: Any | None = None,
        y: Any | None = None,
        *,
        data_config: TabularDataConfig | None = None,
        target_missing_strategy: str | None = None,
        experiment_status_col: ColumnKey | None = None,
        failure_config: ExperimentFailureConfig | None = None,
        cross_validation: bool | None = None,
        **kwargs: Any,
    ) -> Any:
        """Fit objectives once, then attach experiment state and success model."""

        resolved = _resolved_observation_config(
            data_config or self.data_config,
            target_missing_strategy=target_missing_strategy,
            experiment_status_col=experiment_status_col,
        )
        observation_mode = _uses_observation_conversion(resolved)
        run_cv = self.cross_validation if cross_validation is None else bool(cross_validation)
        if observation_mode and run_cv:
            raise ValueError(
                "Cross-validation for partially observed / failed / pending tabular "
                "experiments requires an explicit observation-aware validation protocol."
            )

        configured_failure = (
            self.failure_config if failure_config is None else failure_config
        )
        result = super().fit(
            data=data,
            y=y,
            data_config=resolved,
            cross_validation=False if observation_mode else cross_validation,
            **kwargs,
        )
        self.failure_config = configured_failure

        if isinstance(self.dataset, ObservationTabularDataset):
            observations = self.dataset.observation_data()
            attach_observation_state(
                self.bo,
                observations,
                failure_config=configured_failure,
            )
            self._sync_visualization_metadata()
        return result


__all__ = ["ObservationTabularMixin"]
