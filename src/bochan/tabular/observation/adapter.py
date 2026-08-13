"""Observation-state adapter for the tabular optimizer facade."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable

from bochan.api import ExperimentFailureConfig
from bochan.api.observation.failure import attach_observation_state

from ..config import ColumnKey, TabularDataConfig
from .data import (
    ObservationTabularDataset,
    dataframe_to_observation_tensors,
    numpy_to_observation_tensors,
)


class ObservationAdapter:
    """Resolve observation-aware conversion and attach experiment state."""

    def __init__(
        self,
        failure_config: ExperimentFailureConfig | None = None,
    ) -> None:
        self.failure_config = failure_config

    @staticmethod
    def resolve_config(
        base: TabularDataConfig,
        *,
        target_missing_strategy: str | None,
        experiment_status_col: ColumnKey | None,
    ) -> TabularDataConfig:
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

    @staticmethod
    def uses_observation_conversion(config: TabularDataConfig) -> bool:
        return (
            str(config.target_missing_strategy).strip().lower() == "keep"
            or config.experiment_status_col is not None
        )

    def to_dataset(
        self,
        data: Any,
        y: Any | None,
        *,
        config: TabularDataConfig,
        feature_names: Any = None,
        target_names: Any = None,
        default_converter: Callable[..., Any],
    ) -> Any:
        if not self.uses_observation_conversion(config):
            return default_converter(
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

    def resolve_failure_config(
        self,
        failure_config: ExperimentFailureConfig | None,
    ) -> ExperimentFailureConfig | None:
        return self.failure_config if failure_config is None else failure_config

    def attach(
        self,
        bo: Any,
        dataset: Any,
        *,
        failure_config: ExperimentFailureConfig | None,
    ) -> bool:
        self.failure_config = failure_config
        if not isinstance(dataset, ObservationTabularDataset):
            return False
        attach_observation_state(
            bo,
            dataset.observation_data(),
            failure_config=failure_config,
        )
        return True


__all__ = ["ObservationAdapter"]
