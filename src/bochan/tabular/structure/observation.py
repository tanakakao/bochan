"""Lifecycle hooks for canonical tabular structure-model configuration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any

from bochan.api import ExperimentFailureConfig

from ..config import ColumnKey, TabularDataConfig
from ..observation import ObservationAdapter


class StructureAwareObservationAdapter(ObservationAdapter):
    """Resolve observation config and reapply the structure-model contract.

    ``TabularBayesianOptimizer.fit`` supports fit-time ``data_config`` and
    ``model_config`` overrides. Those overrides are resolved before this adapter
    is called, so this is the common lifecycle point where structure column
    ordering, category maps, backend-specific structure banks, and model policy
    can be validated again without duplicating the core fitter.
    """

    def __init__(
        self,
        owner: Any,
        failure_config: ExperimentFailureConfig | None = None,
    ) -> None:
        super().__init__(failure_config)
        self.owner = owner

    def resolve_config(
        self,
        base: TabularDataConfig,
        *,
        target_missing_strategy: str | None,
        experiment_status_col: ColumnKey | None,
    ) -> TabularDataConfig:
        resolved = super().resolve_config(
            base,
            target_missing_strategy=target_missing_strategy,
            experiment_status_col=experiment_status_col,
        )
        self.owner.source_data_config = resolved
        self.owner.data_config = resolved

        from .m3gnet import configure_tabular_m3gnet

        if not configure_tabular_m3gnet(self.owner):
            from .chgnet import configure_tabular_chgnet

            if not configure_tabular_chgnet(self.owner):
                from .fitting import configure_tabular_alignn

                configure_tabular_alignn(self.owner)
        return self.owner.source_data_config

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
        """Complete mixed metadata and finalize fitted structure-model outputs."""

        if self.owner.structure.enabled:
            completed_bounds = self.owner.structure.complete_categorical_bounds(
                config.bounds,
                data,
                categorical_cols=config.categorical_cols,
                category_maps=config.category_maps,
            )
            config = replace(config, bounds=completed_bounds)
            self.owner.data_config = config

        dataset = super().to_dataset(
            data,
            y,
            config=config,
            feature_names=feature_names,
            target_names=target_names,
            default_converter=default_converter,
        )
        if self.owner.structure.enabled and getattr(dataset, "category_maps", None):
            learned_maps = dict(config.category_maps or {})
            learned_maps.update(dataset.category_maps)
            self.owner.data_config = replace(config, category_maps=learned_maps)

        from .chgnet import configure_chgnet_outputs_from_dataset
        from .m3gnet import validate_m3gnet_outputs_from_dataset
        from .multioutput import configure_alignn_outputs_from_dataset

        validate_m3gnet_outputs_from_dataset(self.owner, dataset)
        configure_chgnet_outputs_from_dataset(self.owner, dataset)
        configure_alignn_outputs_from_dataset(self.owner, dataset)
        return dataset


__all__ = ["StructureAwareObservationAdapter"]
