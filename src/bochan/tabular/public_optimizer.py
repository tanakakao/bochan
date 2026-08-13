"""Canonical pandas / numpy adapter for :class:`bochan.api.BayesianOptimizer`.

Tabular-only concerns are composed explicitly here. Composition bound completion,
composition-total constraints, and variable-total transforms are delegated to
stateless components rather than represented as public optimizer responsibilities.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from bochan.api import OptimizeConfig

from .builders import UNSET
from .composition_bounds_optimizer import CompositionBoundsResolver
from .composition_total_constraints import CompositionTotalConstraintResolver
from .composition_variable_total_transform import CompositionVariableTotalTransform
from .element_column_composition_optimizer import (
    TabularBayesianOptimizer as _ElementColumnTabularBayesianOptimizer,
)
from .element_constraint_composition_optimizer import (
    TabularBayesianOptimizer as _ElementConstraintTabularBayesianOptimizer,
)
from .multi_output_categories import (
    _extract_output_category_maps,
    _merge_target_category_metadata,
)
from .observation_optimizer import ObservationTabularMixin
from .ordinal_rank_labels import (
    resolve_acquisition_ordinal_ranks,
    resolve_ordinal_rank_config,
)
from .prediction import (
    _DATAFRAME_RETURN_TYPES,
    _LABEL_RETURN_TYPES,
    _prediction_tensor_and_index,
    classification_prediction_dataframe,
)


class TabularBayesianOptimizer(
    ObservationTabularMixin,
    _ElementConstraintTabularBayesianOptimizer,
):
    """Single public tabular optimizer delegating BO semantics to the core API."""

    composition_bounds_resolver = CompositionBoundsResolver()
    composition_total_constraint_resolver = CompositionTotalConstraintResolver()
    composition_variable_total_transform = CompositionVariableTotalTransform()

    def __init__(
        self,
        model_config: Any | None = None,
        fit_config: Any | None = None,
        **kwargs: Any,
    ) -> None:
        inferred_maps: dict[Any, dict[Any, int]] = {}

        if isinstance(model_config, Mapping):
            resolved_model_config = dict(model_config)
            multi_output_config = resolved_model_config.get("multi_output_config")
            if multi_output_config is not None:
                resolved_multi_output, maps = _extract_output_category_maps(
                    multi_output_config
                )
                resolved_model_config["multi_output_config"] = resolved_multi_output
                inferred_maps.update(maps)
            model_config = resolved_model_config

        direct_multi_output = kwargs.get("multi_output_config")
        if direct_multi_output is not None:
            resolved_multi_output, maps = _extract_output_category_maps(
                direct_multi_output
            )
            kwargs["multi_output_config"] = resolved_multi_output
            for output_name, category_map in maps.items():
                existing = inferred_maps.get(output_name)
                if existing is not None and existing != category_map:
                    raise ValueError(
                        "Conflicting category declarations for output "
                        f"{output_name!r}."
                    )
                inferred_maps[output_name] = category_map

        _merge_target_category_metadata(kwargs, inferred_maps)
        super().__init__(
            model_config=model_config,
            fit_config=fit_config,
            **kwargs,
        )

    @classmethod
    def _normalize_total_constraints(
        cls,
        constraints: Sequence[Any] | None,
    ) -> list[dict[str, Any]]:
        """Normalize composition total constraints through the owned resolver."""

        return cls.composition_total_constraint_resolver.normalize(constraints)

    def _validate_total_constraints(self) -> None:
        """Validate coupled totals through the owned resolver."""

        self.composition_total_constraint_resolver.validate(
            self.composition_total_constraints,
            self.composition_sites,
        )

    def _named_total_constraints(self) -> list[tuple[Any, ...]]:
        """Translate coupled totals to named model-feature constraints."""

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
        """Merge total constraints through the owned resolver."""

        return cls.composition_total_constraint_resolver.merge_optimize_config(
            opt_config,
            constraints,
        )

    @staticmethod
    def _make_site_search_space(config: Mapping[str, Any]) -> Any:
        """Keep variable-total sites out of fixed-total search-space construction."""

        if config.get("variable_total"):
            return None
        return _ElementColumnTabularBayesianOptimizer._make_site_search_space(config)

    @classmethod
    def _formula_site_totals(
        cls,
        formulas: Any,
        config: Mapping[str, Any],
    ) -> Any:
        """Resolve formula-derived totals through the variable-total component."""

        return cls.composition_variable_total_transform.formula_site_totals(
            formulas,
            config,
        )

    def _site_totals_from_frame(
        self,
        data: Any,
        site_name: str,
        config: Mapping[str, Any],
    ) -> Any:
        """Resolve source-frame totals through the variable-total component."""

        return self.composition_variable_total_transform.site_totals_from_frame(
            data,
            site_name,
            config,
            site_source_columns=self._site_source_columns,
            numeric_site_values=self._numeric_site_values,
        )

    def _prepare_multi_site_frame(
        self,
        data: Any,
        *,
        fit_transformers: bool,
    ) -> Any:
        """Append variable-total features after the element-column base transform."""

        return self.composition_variable_total_transform.prepare_multi_site_frame(
            data,
            fit_transformers=fit_transformers,
            composition_sites=self.composition_sites,
            base_prepare=lambda frame, *, fit_transformers: (
                _ElementColumnTabularBayesianOptimizer._prepare_multi_site_frame(
                    self,
                    frame,
                    fit_transformers=fit_transformers,
                )
            ),
            site_source_columns=self._site_source_columns,
            numeric_site_values=self._numeric_site_values,
        )

    def _replace_multi_site_input_cols(
        self,
        input_cols: Sequence[Any] | None,
    ) -> list[Any] | None:
        """Map source composition columns to coordinates and total features."""

        return self.composition_variable_total_transform.replace_input_cols(
            input_cols,
            composition_sites=self.composition_sites,
            composition_transformers=self.composition_transformers_,
            site_source_columns=self._site_source_columns,
        )

    def _expanded_bounds(self, bounds: Any, transformed: Any) -> Any:
        """Complete transformed single-site composition bounds."""

        expanded = super()._expanded_bounds(bounds, transformed)
        return self.composition_bounds_resolver.complete(expanded, transformed)

    def _expanded_multi_site_bounds(self, bounds: Any, transformed: Any) -> Any:
        """Complete base, variable-total, and generic transformed bounds."""

        expanded = _ElementColumnTabularBayesianOptimizer._expanded_multi_site_bounds(
            self,
            bounds,
            transformed,
        )
        expanded = self.composition_variable_total_transform.complete_bounds(
            expanded,
            composition_sites=self.composition_sites,
            composition_transformers=self.composition_transformers_,
        )
        return self.composition_bounds_resolver.complete(expanded, transformed)

    @classmethod
    def _dynamic_search_space(
        cls,
        config: Mapping[str, Any],
        total: float,
    ) -> Any:
        """Build candidate-specific repair bounds through the transform component."""

        return cls.composition_variable_total_transform.dynamic_search_space(
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
        """Restore variable totals before applying element-constraint projection."""

        restored = _ElementColumnTabularBayesianOptimizer.inverse_compositions(
            self,
            data,
            repair=repair,
            keep_coordinates=keep_coordinates,
        )
        restored = self.composition_variable_total_transform.inverse_compositions(
            data,
            restored,
            composition_sites=self.composition_sites,
            composition_transformers=self.composition_transformers_,
            multi_site_composition_enabled=self.multi_site_composition_enabled,
            repair=repair,
        )
        if (
            repair
            and self.multi_site_composition_enabled
            and self.composition_element_constraints
        ):
            restored = self._repair_element_constraint_frame(restored)
        return restored

    def candidate(
        self,
        acq_config: Any | None = None,
        opt_config: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        """Resolve tabular ordinal labels before delegating candidate generation."""

        target_names = list(self.dataset.target_names) if self.dataset is not None else []
        target_category_maps = (
            dict(getattr(self.dataset, "target_category_maps", None) or {})
            if self.dataset is not None
            else {}
        )
        if target_names:
            acq_config = resolve_acquisition_ordinal_ranks(
                acq_config,
                target_names=target_names,
                target_category_maps=target_category_maps,
            )
            outcome_constraint_config = kwargs.get(
                "outcome_constraint_config",
                UNSET,
            )
            if outcome_constraint_config is not UNSET:
                kwargs["outcome_constraint_config"] = resolve_ordinal_rank_config(
                    outcome_constraint_config,
                    target_names=target_names,
                    target_category_maps=target_category_maps,
                )
        return super().candidate(
            acq_config=acq_config,
            opt_config=opt_config,
            **kwargs,
        )

    def predict(
        self,
        data: Any,
        *,
        return_type: str = "dataframe",
        include_input: bool = False,
        return_dataframe_input: bool = False,
        posterior_kwargs: dict[str, Any] | None = None,
        include_prediction_labels: bool = True,
        binary_threshold: float = 0.5,
        **kwargs: Any,
    ) -> Any:
        """Predict and optionally append decoded classification labels."""

        normalized_return_type = str(return_type).lower()
        labels_only = normalized_return_type in _LABEL_RETURN_TYPES
        dataframe_return = normalized_return_type in _DATAFRAME_RETURN_TYPES

        if not labels_only:
            result = super().predict(
                data,
                return_type=return_type,
                include_input=include_input,
                return_dataframe_input=return_dataframe_input,
                posterior_kwargs=posterior_kwargs,
                **kwargs,
            )
            if not include_prediction_labels or not dataframe_return:
                return result
            if return_dataframe_input:
                prediction_df, returned_input = result
            else:
                prediction_df = result
                returned_input = None
        else:
            prediction_df = None
            returned_input = data if return_dataframe_input else None

        X, original_index = _prediction_tensor_and_index(self, data)
        labels_df = classification_prediction_dataframe(
            self,
            X,
            posterior_kwargs=posterior_kwargs,
            binary_threshold=binary_threshold,
        )
        if labels_only and labels_df.shape[1] == 0:
            raise ValueError(
                "The fitted optimizer has no binary, multiclass, or ordinal outputs."
            )
        if original_index is not None:
            labels_df.index = original_index

        if prediction_df is None:
            output_df = labels_df
            if include_input:
                input_df = self._prediction_input_to_dataframe(data, X)
                if original_index is not None:
                    input_df.index = original_index
                output_df = input_df.join(output_df)
        elif labels_df.shape[1] == 0:
            output_df = prediction_df
        else:
            attrs = dict(getattr(prediction_df, "attrs", {}) or {})
            output_df = prediction_df.join(labels_df)
            output_df.attrs.update(attrs)

        if return_dataframe_input:
            return output_df, returned_input
        return output_df


__all__ = ["TabularBayesianOptimizer"]
