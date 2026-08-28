"""Candidate-generation orchestration for the tabular optimizer facade."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import fields, replace
from time import perf_counter
from typing import Any

from bochan.api import AcquisitionConfig, DataContext, OptimizeConfig
from bochan.api.progress import emit_progress

from ..composition.constraints import (
    CompositionElementConstraintCandidateReranker,
    CompositionElementConstraintProjector,
    CompositionElementConstraintResolver,
)
from ..composition.totals import CompositionTotalConstraintResolver
from ..config import make_acquisition_config, make_optimize_config
from ..data import resolve_column_indices, resolve_dtype, resolve_optimize_config_columns
from ..targets import (
    resolve_acquisition_config_columns,
    resolve_acquisition_ordinal_ranks,
    resolve_ordinal_rank_config,
    resolve_outcome_constraint_config_columns,
)

_ACQ_KEYS = {field.name for field in fields(AcquisitionConfig)} | {
    "acq_name",
    "objective_mode",
    "objective_output",
    "objective_outputs",
    "objective_specs",
    "objective_directions",
    "objective_weights",
    "objective_eq_targets",
    "objective_direction",
    "objective_weight",
    "objective_eq_target",
    "objective_n_w",
    "objective_risk_type",
    "objective_alpha",
    "objective_maximize",
    "objective_aggregate_mean_when_no_risk",
    "objective_allow_unexpanded",
    "objective_utility_values",
    "objective_ordinal_likelihood",
}
_OPT_KEYS = {field.name for field in fields(OptimizeConfig)} | {
    "constraints",
    "repair_bounds",
    "numeric_indices",
    "steps",
    "comp_idx",
    "k",
    "repair_equality_constraints",
    "repair_inequality_constraints",
    "repair_inequality_sense",
    "repair_fixed_features",
    "final_sum_constraint",
    "support_selection",
    "sample_tau",
    "sample_eps",
    "generator",
    "max_iters",
    "num_alternations",
    "final_priority",
    "support_eps",
}


def _take(values: dict[str, Any], keys: set[str]) -> dict[str, Any]:
    return {key: values.pop(key) for key in list(values) if key in keys}


def _resolve_named_objective_outputs(
    values: Mapping[str, Any],
    target_names: Sequence[Any],
) -> dict[str, Any]:
    """Resolve explicitly supplied tabular target names to output indices.

    Omitted objective fields are intentionally left untouched so single-output
    models keep the core API's existing implicit objective selection.
    """

    resolved = dict(values)

    def resolve_one(value: Any) -> Any:
        if value is None:
            return None
        indices = resolve_column_indices(value, target_names)
        if not indices:
            return value
        return int(indices[0])

    def resolve_many(value: Any) -> Any:
        if value is None:
            return None
        indices = resolve_column_indices(value, target_names)
        return None if indices is None else [int(index) for index in indices]

    if "objective_output" in resolved:
        resolved["objective_output"] = resolve_one(resolved["objective_output"])
    if "objective_outputs" in resolved:
        resolved["objective_outputs"] = resolve_many(resolved["objective_outputs"])

    objective_config = resolved.get("objective_config")
    if isinstance(objective_config, Mapping):
        nested = dict(objective_config)
        if "output" in nested:
            nested["output"] = resolve_one(nested["output"])
        if "outputs" in nested:
            nested["outputs"] = resolve_many(nested["outputs"])
        resolved["objective_config"] = nested

    objective_specs = resolved.get("objective_specs")
    if objective_specs is not None:
        specs = []
        for spec in objective_specs:
            if isinstance(spec, Mapping) and "output" in spec:
                spec = dict(spec)
                spec["output"] = resolve_one(spec["output"])
            specs.append(spec)
        resolved["objective_specs"] = specs

    return resolved


class CandidateService:
    """Prepare tabular configs and postprocess composition candidates."""

    def __init__(
        self,
        *,
        composition: Any,
        total_constraints: Sequence[Any] | None,
        element_constraints: Sequence[Any] | None,
        rerank: bool,
        rerank_factor: int,
        max_supports: int,
    ) -> None:
        self.composition = composition
        self.total_resolver = CompositionTotalConstraintResolver()
        self.element_resolver = CompositionElementConstraintResolver()
        self.reranker = CompositionElementConstraintCandidateReranker()
        self.total_constraints = self.total_resolver.normalize(total_constraints)
        self.element_constraints = self.element_resolver.normalize(element_constraints)
        self.rerank_enabled = bool(rerank)
        self.rerank_factor = int(rerank_factor)
        self.max_supports = int(max_supports)
        if self.rerank_factor < 1:
            raise ValueError("composition_constraint_rerank_factor must be >= 1.")
        if self.max_supports < 1:
            raise ValueError("composition_constraint_max_supports must be >= 1.")
        self.total_resolver.validate(self.total_constraints, composition.sites)
        self.projector().validate()

    def projector(self) -> CompositionElementConstraintProjector:
        return CompositionElementConstraintProjector(
            composition_sites=self.composition.sites,
            composition_element_constraints=self.element_constraints,
            composition_transformers=self.composition.transformers,
            max_supports=self.max_supports,
        )

    def repair_compositions(
        self,
        restored: Any,
        *,
        repair: bool,
    ) -> Any:
        if repair and self.composition.enabled and self.element_constraints:
            return self.projector().repair_frame(restored)
        return restored

    def _prepare_configs(
        self,
        owner: Any,
        acq_config: Any,
        opt_config: Any,
        values: dict[str, Any],
    ) -> tuple[Any, OptimizeConfig]:
        target_names = list(owner.dataset.target_names) if owner.dataset is not None else []
        target_maps = (
            dict(getattr(owner.dataset, "target_category_maps", None) or {})
            if owner.dataset is not None
            else {}
        )
        if target_names:
            acq_config = resolve_acquisition_config_columns(
                acq_config,
                target_names,
                target_maps,
            )
            if isinstance(acq_config, Mapping):
                acq_config = _resolve_named_objective_outputs(acq_config, target_names)
            acq_config = resolve_acquisition_ordinal_ranks(
                acq_config,
                target_names=target_names,
                target_category_maps=target_maps,
            )
            if "outcome_constraint_config" in values:
                outcome = resolve_outcome_constraint_config_columns(
                    values["outcome_constraint_config"],
                    target_names,
                    target_maps,
                )
                values["outcome_constraint_config"] = resolve_ordinal_rank_config(
                    outcome,
                    target_names=target_names,
                    target_category_maps=target_maps,
                )

        acq_values = _take(values, _ACQ_KEYS)
        if target_names and acq_values:
            acq_values = _resolve_named_objective_outputs(acq_values, target_names)
        if isinstance(acq_config, Mapping) or acq_values:
            acq_config = make_acquisition_config(acq_config, **acq_values)

        opt_values = _take(values, _OPT_KEYS)
        if isinstance(opt_config, Mapping) or opt_values:
            opt_config = make_optimize_config(opt_config, **opt_values)
        elif opt_config is None:
            opt_config = OptimizeConfig()

        total_constraints = self.total_resolver.named_constraints(
            self.total_constraints,
            self.composition.sites,
        )
        opt_config = self.total_resolver.merge_optimize_config(
            opt_config,
            total_constraints,
        )
        element_constraints = self.element_resolver.named_constraints(
            self.element_constraints,
            self.composition.sites,
            self.composition.transformers,
        )
        opt_config = self.total_resolver.merge_optimize_config(
            opt_config,
            element_constraints,
        )
        if isinstance(opt_config, Mapping):
            opt_config = make_optimize_config(opt_config)
        if values:
            raise TypeError(f"Unknown candidate arguments: {sorted(values)!r}.")
        return acq_config, opt_config

    @staticmethod
    def _raw_candidate(
        owner: Any,
        acq_config: AcquisitionConfig,
        opt_config: OptimizeConfig,
        *,
        data_context: DataContext | None,
        bounds: Any,
        return_result: bool,
    ) -> Any:
        resolved_opt = resolve_optimize_config_columns(
            opt_config,
            owner.dataset.feature_names,
            dtype=resolve_dtype(owner.data_config.dtype),
            device=owner.data_config.device,
        )
        started = perf_counter()
        emit_progress(
            "candidate_generation_started",
            q=int(resolved_opt.q),
            sequential=bool(resolved_opt.sequential),
            optimizer=str(resolved_opt.optimizer),
        )
        try:
            result = owner.bo.candidate(
                acq_config,
                resolved_opt,
                data_context=data_context,
                bounds=owner.dataset.bounds if bounds is None else bounds,
                return_result=return_result,
            )
        except Exception:
            emit_progress(
                "candidate_generation_failed",
                q=int(resolved_opt.q),
                duration_ms=round((perf_counter() - started) * 1000, 3),
            )
            raise
        emit_progress(
            "candidate_generation_completed",
            q=int(resolved_opt.q),
            duration_ms=round((perf_counter() - started) * 1000, 3),
        )
        return result

    def generate(
        self,
        owner: Any,
        acq_config: AcquisitionConfig | Mapping[str, Any] | None,
        opt_config: OptimizeConfig | Mapping[str, Any] | None,
        *,
        data_context: DataContext | None,
        bounds: Any,
        return_dataframe: bool,
        return_result: bool,
        return_composition: bool,
        keep_composition_coordinates: bool,
        composition_constraint_rerank: bool | None,
        composition_constraint_rerank_factor: int | None,
        values: dict[str, Any],
    ) -> Any:
        owner._check_fitted()
        acq_config, opt_config = self._prepare_configs(
            owner,
            acq_config,
            opt_config,
            values,
        )
        if acq_config is None:
            raise ValueError("acq_name or acq_config is required.")

        descriptor_sites = self.composition.descriptor_sites()
        if self.composition.enabled and descriptor_sites:
            raise ValueError(
                "Composition descriptors are supported for fit/predict but are not "
                f"independent decision variables. Disable them at {descriptor_sites!r}."
            )

        rerank = (
            self.rerank_enabled
            if composition_constraint_rerank is None
            else bool(composition_constraint_rerank)
        )
        if (
            self.element_constraints
            and rerank
            and not return_result
            and return_dataframe
            and return_composition
        ):
            requested_q = self.reranker.requested_q(opt_config, {})
            factor = (
                self.rerank_factor
                if composition_constraint_rerank_factor is None
                else int(composition_constraint_rerank_factor)
            )
            if factor < 1:
                raise ValueError("composition_constraint_rerank_factor must be >= 1.")
            result = self._raw_candidate(
                owner,
                acq_config,
                replace(opt_config, q=requested_q * factor),
                data_context=data_context,
                bounds=bounds,
                return_result=True,
            )
            raw = owner.candidates_to_dataframe(result.candidates)
            repaired = owner.inverse_compositions(
                raw,
                repair=True,
                keep_coordinates=keep_composition_coordinates,
            )
            try:
                return self.reranker.rerank(
                    repaired,
                    result.acqf,
                    requested_q,
                    transform_compositions=owner.transform_compositions,
                    data_config=owner.data_config,
                    feature_names=owner.dataset.feature_names,
                )
            except (RuntimeError, ValueError, TypeError, KeyError):
                selected = repaired.drop_duplicates().head(requested_q).reset_index(drop=True)
                return selected, result.acq_value

        result = self._raw_candidate(
            owner,
            acq_config,
            opt_config,
            data_context=data_context,
            bounds=bounds,
            return_result=return_result,
        )
        if return_result:
            return result
        candidates, acq_value = result
        if not return_dataframe:
            return candidates, acq_value
        frame = owner.candidates_to_dataframe(candidates)
        if self.composition.enabled and return_composition:
            frame = owner.inverse_compositions(
                frame,
                repair=True,
                keep_coordinates=keep_composition_coordinates,
            )
        return frame, acq_value


__all__ = ["CandidateService"]
