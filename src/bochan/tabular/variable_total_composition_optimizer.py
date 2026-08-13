"""Variable-total adapter for multi-site composition optimization."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from bochan.api import OptimizeConfig

from .composition import CompositionSearchSpace
from .composition_total_constraints import CompositionTotalConstraintResolver
from .composition_variable_total import CompositionVariableTotalTransformer
from .element_column_composition_optimizer import (
    TabularBayesianOptimizer as _ElementColumnTabularBayesianOptimizer,
)


class TabularBayesianOptimizer(_ElementColumnTabularBayesianOptimizer):
    """Thin adapter composing variable-total transformation components."""

    composition_total_constraint_resolver = CompositionTotalConstraintResolver()
    composition_variable_total_transformer = CompositionVariableTotalTransformer()

    def __init__(
        self,
        model_config: Any | None = None,
        fit_config: Any | None = None,
        *,
        composition_sites: Mapping[str, Mapping[str, Any]] | None = None,
        composition_total_constraints: Sequence[Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self.composition_total_constraints = (
            self.composition_total_constraint_resolver.normalize(
                composition_total_constraints
            )
        )
        super().__init__(
            model_config=model_config,
            fit_config=fit_config,
            composition_sites=composition_sites,
            **kwargs,
        )
        self._validate_total_constraints()

    @classmethod
    def _normalize_composition_sites(
        cls,
        sites: Mapping[str, Mapping[str, Any]] | None,
    ) -> dict[str, dict[str, Any]]:
        return cls.composition_variable_total_transformer.normalize_sites(
            sites,
            base_normalizer=(
                _ElementColumnTabularBayesianOptimizer._normalize_composition_sites
            ),
        )

    @classmethod
    def _normalize_total_constraints(
        cls,
        constraints: Sequence[Any] | None,
    ) -> list[dict[str, Any]]:
        return cls.composition_total_constraint_resolver.normalize(constraints)

    def _validate_total_constraints(self) -> None:
        self.composition_total_constraint_resolver.validate(
            self.composition_total_constraints,
            self.composition_sites,
        )

    @classmethod
    def _make_site_search_space(
        cls,
        config: Mapping[str, Any],
    ) -> CompositionSearchSpace | None:
        return cls.composition_variable_total_transformer.make_site_search_space(
            config,
            base_factory=_ElementColumnTabularBayesianOptimizer._make_site_search_space,
        )

    @classmethod
    def _formula_site_totals(
        cls,
        formulas: Any,
        config: Mapping[str, Any],
    ) -> np.ndarray:
        return cls.composition_variable_total_transformer.formula_site_totals(
            formulas,
            config,
        )

    def _site_totals_from_frame(
        self,
        data: Any,
        site_name: str,
        config: Mapping[str, Any],
    ) -> np.ndarray | None:
        return self.composition_variable_total_transformer.site_totals_from_frame(
            self,
            data,
            site_name,
            config,
        )

    def _prepare_multi_site_frame(
        self,
        data: Any,
        *,
        fit_transformers: bool,
    ) -> Any:
        totals = self.composition_variable_total_transformer.collect_totals(self, data)
        transformed = super()._prepare_multi_site_frame(
            data,
            fit_transformers=fit_transformers,
        )
        return self.composition_variable_total_transformer.inject_totals(
            self,
            transformed,
            totals,
        )

    def _replace_multi_site_input_cols(
        self,
        input_cols: Sequence[Any] | None,
    ) -> list[Any] | None:
        return self.composition_variable_total_transformer.replace_input_cols(
            self,
            input_cols,
        )

    def _expanded_multi_site_bounds(self, bounds: Any, transformed: Any) -> Any:
        expanded = super()._expanded_multi_site_bounds(bounds, transformed)
        return self.composition_variable_total_transformer.complete_bounds(
            self,
            expanded,
        )

    def _named_total_constraints(self) -> list[tuple[Any, ...]]:
        return self.composition_total_constraint_resolver.named_constraints(
            self.composition_total_constraints,
            self.composition_sites,
        )

    @classmethod
    def _merge_total_constraints(
        cls,
        opt_config: OptimizeConfig | Mapping[str, Any] | None,
        constraints: Sequence[tuple[Any, ...]],
    ) -> OptimizeConfig | Mapping[str, Any] | None:
        return cls.composition_total_constraint_resolver.merge_optimize_config(
            opt_config,
            constraints,
        )

    def candidate(
        self,
        acq_config: Any | None = None,
        opt_config: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        named_constraints = self._named_total_constraints()
        opt_config = self._merge_total_constraints(opt_config, named_constraints)
        return super().candidate(
            acq_config=acq_config,
            opt_config=opt_config,
            **kwargs,
        )

    @classmethod
    def _dynamic_search_space(
        cls,
        config: Mapping[str, Any],
        total: float,
    ) -> CompositionSearchSpace:
        return cls.composition_variable_total_transformer.dynamic_search_space(
            config,
            total,
        )

    def inverse_compositions(
        self,
        data: Any,
        *,
        repair: bool = True,
        keep_coordinates: bool = False,
    ) -> Any:
        restored = super().inverse_compositions(
            data,
            repair=repair,
            keep_coordinates=keep_coordinates,
        )
        return self.composition_variable_total_transformer.restore(
            self,
            data,
            restored,
            repair=repair,
        )


__all__ = ["TabularBayesianOptimizer"]
