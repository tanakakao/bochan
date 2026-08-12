"""Canonical pandas / numpy adapter for :class:`bochan.api.BayesianOptimizer`.

Tabular-only concerns are composed explicitly here. Composition bound completion
is delegated to a stateless resolver rather than represented by another optimizer
subclass in the inheritance chain.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .builders import UNSET
from .composition_bounds_optimizer import CompositionBoundsResolver
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
from .prediction_labels import (
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

    def _expanded_bounds(self, bounds: Any, transformed: Any) -> Any:
        """Complete transformed single-site composition bounds."""

        expanded = super()._expanded_bounds(bounds, transformed)
        return self.composition_bounds_resolver.complete(expanded, transformed)

    def _expanded_multi_site_bounds(self, bounds: Any, transformed: Any) -> Any:
        """Complete transformed multi-site composition bounds."""

        expanded = super()._expanded_multi_site_bounds(bounds, transformed)
        return self.composition_bounds_resolver.complete(expanded, transformed)

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
