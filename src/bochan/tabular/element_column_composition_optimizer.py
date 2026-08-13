"""Ordering adapter for element-column multi-site composition inputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .composition_element_columns import CompositionElementColumnTransform
from .multi_site_composition_optimizer import (
    TabularBayesianOptimizer as _FormulaMultiSiteTabularBayesianOptimizer,
)


class TabularBayesianOptimizer(_FormulaMultiSiteTabularBayesianOptimizer):
    """Order element-column conversion around the formula multi-site workflow."""

    composition_element_column_transform = CompositionElementColumnTransform()

    @classmethod
    def _normalize_composition_sites(
        cls,
        sites: Mapping[str, Mapping[str, Any]] | None,
    ) -> dict[str, dict[str, Any]]:
        return cls.composition_element_column_transform.normalize_sites(
            sites,
            base_normalizer=(_FormulaMultiSiteTabularBayesianOptimizer._normalize_composition_sites),
        )

    def _prepare_multi_site_frame(
        self,
        data: Any,
        *,
        fit_transformers: bool,
    ) -> Any:
        return self.composition_element_column_transform.prepare_multi_site_frame(
            data,
            fit_transformers=fit_transformers,
            composition_sites=self.composition_sites,
            composition_transformers=self.composition_transformers_,
            base_prepare=lambda frame, *, fit_transformers: super(
                TabularBayesianOptimizer, self
            )._prepare_multi_site_frame(
                frame,
                fit_transformers=fit_transformers,
            ),
        )

    def _replace_multi_site_input_cols(
        self,
        input_cols: Sequence[Any] | None,
    ) -> list[Any] | None:
        return self.composition_element_column_transform.replace_input_cols(
            input_cols,
            composition_sites=self.composition_sites,
            composition_transformers=self.composition_transformers_,
        )

    def _expanded_multi_site_bounds(self, bounds: Any, transformed: Any) -> Any:
        expanded = super()._expanded_multi_site_bounds(bounds, transformed)
        return self.composition_element_column_transform.clean_bounds(
            expanded,
            composition_sites=self.composition_sites,
        )

    def fit(
        self,
        data: Any | None = None,
        y: Any | None = None,
        *,
        input_cols: Sequence[Any] | None = None,
        categorical_cols: Sequence[Any] | None = None,
        bounds: Any = None,
        **kwargs: Any,
    ) -> TabularBayesianOptimizer:
        if not self.multi_site_composition_enabled:
            return super().fit(
                data=data,
                y=y,
                input_cols=input_cols,
                categorical_cols=categorical_cols,
                bounds=bounds,
                **kwargs,
            )
        resolved_categorical = self.composition_element_column_transform.resolve_categorical_cols(
            categorical_cols,
            default_categorical_cols=self.data_config.categorical_cols or (),
            composition_sites=self.composition_sites,
        )
        return super().fit(
            data=data,
            y=y,
            input_cols=input_cols,
            categorical_cols=resolved_categorical,
            bounds=bounds,
            **kwargs,
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
        return self.composition_element_column_transform.inverse_compositions(
            restored,
            composition_sites=self.composition_sites,
            composition_transformers=self.composition_transformers_,
            enabled=self.multi_site_composition_enabled,
        )


__all__ = ["TabularBayesianOptimizer"]
