"""Canonical pandas / numpy adapter for :class:`bochan.api.BayesianOptimizer`.

Tabular-only concerns are composed explicitly here. Composition transforms,
bounds, and constraints are delegated to stateless components rather than
represented as public optimizer inheritance layers.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from bochan.api import OptimizeConfig

from .builders import UNSET
from .composition_bounds_optimizer import CompositionBoundsResolver
from .composition_element_constraint_candidates import (
    CompositionElementConstraintCandidateReranker,
)
from .composition_element_constraints import (
    CompositionElementConstraintProjector,
    CompositionElementConstraintResolver,
)
from .composition_total_constraints import CompositionTotalConstraintResolver
from .composition_variable_total_transform import CompositionVariableTotalTransform
from .element_column_composition_optimizer import (
    TabularBayesianOptimizer as _ElementColumnTabularBayesianOptimizer,
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
    _ElementColumnTabularBayesianOptimizer,
):
    """Single public tabular optimizer delegating BO semantics to the core API."""

    composition_bounds_resolver = CompositionBoundsResolver()
    composition_total_constraint_resolver = CompositionTotalConstraintResolver()
    composition_variable_total_transform = CompositionVariableTotalTransform()
    composition_element_constraint_resolver = CompositionElementConstraintResolver()
    composition_element_constraint_candidate_reranker = (
        CompositionElementConstraintCandidateReranker()
    )

    def __init__(
        self,
        model_config: Any | None = None,
        fit_config: Any | None = None,
        *,
        composition_sites: Mapping[str, Mapping[str, Any]] | None = None,
        composition_total_constraints: Sequence[Any] | None = None,
        composition_element_constraints: Sequence[Any] | None = None,
        composition_constraint_rerank: bool = True,
        composition_constraint_rerank_factor: int = 4,
        composition_constraint_max_supports: int = 256,
        **kwargs: Any,
    ) -> None:
        self.composition_total_constraints = (
            self.composition_total_constraint_resolver.normalize(
                composition_total_constraints
            )
        )
        self.composition_element_constraints = (
            self.composition_element_constraint_resolver.normalize(
                composition_element_constraints
            )
        )
        self.composition_constraint_rerank = bool(composition_constraint_rerank)
        self.composition_constraint_rerank_factor = int(
            composition_constraint_rerank_factor
        )
        self.composition_constraint_max_supports = int(
            composition_constraint_max_supports
        )
        if self.composition_constraint_rerank_factor < 1:
            raise ValueError("composition_constraint_rerank_factor must be >= 1.")
        if self.composition_constraint_max_supports < 1:
            raise ValueError("composition_constraint_max_supports must be >= 1.")

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
            composition_sites=composition_sites,
            **kwargs,
        )
        self._validate_total_constraints()
        self._make_element_constraint_projector().validate()

    @classmethod
    def _normalize_composition_sites(
        cls,
        sites: Mapping[str, Mapping[str, Any]] | None,
    ) -> dict[str, dict[str, Any]]:
        """Normalize variable-total sites through the transform component."""

        return cls.composition_variable_total_transform.normalize_sites(
            sites,
            base_normalizer=(
                _ElementColumnTabularBayesianOptimizer._normalize_composition_sites
            ),
        )

    @classmethod
    def _make_site_search_space(
        cls,
        config: Mapping[str, Any],
    ) -> Any:
        """Use dynamic search spaces for variable-total sites."""

        return cls.composition_variable_total_transform.make_site_search_space(
            config,
            base_factory=_ElementColumnTabularBayesianOptimizer._make_site_search_space,
        )

    @classmethod
    def _formula_site_totals(
        cls,
        formulas: Any,
        config: Mapping[str, Any],
    ) -> Any:
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
        """Inject variable-total features around the element-column transform."""

        return self.composition_variable_total_transform.prepare_multi_site_frame(
            data,
            fit_transformers=fit_transformers,
            composition_sites=self.composition_sites,
            base_prepare=lambda frame, *, fit_transformers: super(
                TabularBayesianOptimizer, self
            )._prepare_multi_site_frame(
                frame,
                fit_transformers=fit_transformers,
            ),
            site_source_columns=self._site_source_columns,
            numeric_site_values=self._numeric_site_values,
        )

    def _replace_multi_site_input_cols(
        self,
        input_cols: Sequence[Any] | None,
    ) -> list[Any] | None:
        return self.composition_variable_total_transform.replace_input_cols(
            input_cols,
            composition_sites=self.composition_sites,
            composition_transformers=self.composition_transformers_,
            site_source_columns=self._site_source_columns,
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

    def _make_element_constraint_projector(
        self,
    ) -> CompositionElementConstraintProjector:
        """Build the projector from the current fitted composition context."""

        return CompositionElementConstraintProjector(
            composition_sites=self.composition_sites,
            composition_element_constraints=self.composition_element_constraints,
            composition_transformers=getattr(self, "composition_transformers_", {}),
            max_supports=self.composition_constraint_max_supports,
        )

    def _named_element_constraints(self) -> list[tuple[Any, ...]]:
        """Translate compatible element constraints to model coordinates."""

        return self.composition_element_constraint_resolver.named_constraints(
            self.composition_element_constraints,
            self.composition_sites,
            self.composition_transformers_,
        )

    def inverse_compositions(
        self,
        data: Any,
        *,
        repair: bool = True,
        keep_coordinates: bool = False,
    ) -> Any:
        """Restore variable totals before applying element-constraint repair."""

        restored = super().inverse_compositions(
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
            restored = self._make_element_constraint_projector().repair_frame(restored)
        return restored

    def _expanded_bounds(self, bounds: Any, transformed: Any) -> Any:
        """Complete transformed single-site composition bounds."""

        expanded = super()._expanded_bounds(bounds, transformed)
        return self.composition_bounds_resolver.complete(expanded, transformed)

    def _expanded_multi_site_bounds(self, bounds: Any, transformed: Any) -> Any:
        """Complete variable-total and transformed composition bounds."""

        expanded = super()._expanded_multi_site_bounds(bounds, transformed)
        expanded = self.composition_variable_total_transform.complete_bounds(
            expanded,
            composition_sites=self.composition_sites,
            composition_transformers=self.composition_transformers_,
        )
        return self.composition_bounds_resolver.complete(expanded, transformed)

    def candidate(
        self,
        acq_config: Any | None = None,
        opt_config: Any | None = None,
        *,
        return_dataframe: bool = True,
        return_result: bool = False,
        return_composition: bool = True,
        keep_composition_coordinates: bool = False,
        composition_constraint_rerank: bool | None = None,
        composition_constraint_rerank_factor: int | None = None,
        **kwargs: Any,
    ) -> Any:
        """Resolve tabular constraints and return repaired candidate compositions."""

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

        opt_config = self._merge_total_constraints(
            opt_config,
            self._named_total_constraints(),
        )
        opt_config = CompositionTotalConstraintResolver.merge_optimize_config(
            opt_config,
            self._named_element_constraints(),
        )

        rerank = (
            self.composition_constraint_rerank
            if composition_constraint_rerank is None
            else bool(composition_constraint_rerank)
        )
        if (
            not self.composition_element_constraints
            or not rerank
            or return_result
            or not return_dataframe
            or not return_composition
        ):
            return super().candidate(
                acq_config=acq_config,
                opt_config=opt_config,
                return_dataframe=return_dataframe,
                return_result=return_result,
                return_composition=return_composition,
                keep_composition_coordinates=keep_composition_coordinates,
                **kwargs,
            )

        requested_q = (
            self.composition_element_constraint_candidate_reranker.requested_q(
                opt_config,
                kwargs,
            )
        )
        factor = (
            self.composition_constraint_rerank_factor
            if composition_constraint_rerank_factor is None
            else int(composition_constraint_rerank_factor)
        )
        if factor < 1:
            raise ValueError("composition_constraint_rerank_factor must be >= 1.")

        call_kwargs = dict(kwargs)
        call_kwargs["q"] = requested_q * factor
        result = super().candidate(
            acq_config=acq_config,
            opt_config=opt_config,
            return_dataframe=True,
            return_result=True,
            return_composition=False,
            **call_kwargs,
        )
        raw_candidates = self.candidates_to_dataframe(result.candidates)
        repaired = self.inverse_compositions(
            raw_candidates,
            repair=True,
            keep_coordinates=keep_composition_coordinates,
        )
        try:
            return self.composition_element_constraint_candidate_reranker.rerank(
                repaired,
                result.acqf,
                requested_q,
                transform_compositions=self.transform_compositions,
                data_config=self.data_config,
                feature_names=self.dataset.feature_names,
            )
        except (RuntimeError, ValueError, TypeError, KeyError):
            selected = repaired.drop_duplicates().head(requested_q).reset_index(drop=True)
            return selected, result.acq_value

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
