"""Structure-aware candidate generation for tabular ALIGNN models."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from ..optimizer.candidates import CandidateService
from .scaling import optimize_alignn_structure_alternating

_ALIGNN_MODEL_TYPES = frozenset({"alignn_gp", "alignn_dkl"})
_ALTERNATING_STRUCTURE_THRESHOLD = 10
_ALTERNATING_OPTION_KEYS = frozenset(
    {
        "initialization_strategy",
        "tol",
        "maxiter_alternating",
        "maxiter_discrete",
        "maxiter_continuous",
        "maxiter_init",
        "max_discrete_values",
        "num_spray_points",
        "std_cont_perturbation",
        "batch_limit",
        "init_batch_limit",
    }
)
_ALTERNATING_OPTIMIZER_KWARGS = frozenset({"options", "alternating_options"})


def _process_category_fixed_features(owner: Any) -> list[dict[int, float]]:
    """Return current joint process-category assignments for mixed ALIGNN."""

    if owner.bo.bundle is None:
        return [{}]
    cat_dims = [int(index) for index in (owner.bo.bundle.cat_dims or [])]
    if not cat_dims:
        return [{}]
    if 0 in cat_dims:
        raise RuntimeError(
            "ALIGNN model cat_dims must contain process categories only; "
            "structure feature 0 is enumerated separately."
        )

    train_X = owner.bo.train_X
    if train_X is None:
        raise RuntimeError("No current ALIGNN training inputs are available for optimization.")

    import torch

    values = train_X[:, cat_dims]
    unique_rows = torch.unique(values, dim=0)
    if unique_rows.numel() == 0:
        raise RuntimeError("No categorical process assignments are available for optimization.")
    return [
        {dim: float(value) for dim, value in zip(cat_dims, row, strict=True)}
        for row in unique_rows.detach().cpu().tolist()
    ]


def _optimizer_name(value: Any) -> str | None:
    if callable(value) and not isinstance(value, str):
        return None
    return str(value).replace("-", "_").lower()


def _alternating_optimizer_kwargs_compatible(resolved_opt: Any) -> bool:
    """Reject standard optimizer kwargs that have no alternating equivalent."""

    kwargs = dict(getattr(resolved_opt, "optimizer_kwargs", None) or {})
    if set(kwargs) - _ALTERNATING_OPTIMIZER_KWARGS:
        return False
    for name in _ALTERNATING_OPTIMIZER_KWARGS:
        options = kwargs.get(name)
        if options is None:
            continue
        if not isinstance(options, Mapping):
            return False
        if set(options) - _ALTERNATING_OPTION_KEYS:
            return False
    return True


def _use_alternating_structure_search(
    resolved_opt: Any,
    *,
    structure_count: int,
) -> bool:
    """Return whether Phase-6 structure scaling should replace full enumeration."""

    if structure_count <= _ALTERNATING_STRUCTURE_THRESHOLD:
        return False
    if int(resolved_opt.q) != 1 or not bool(resolved_opt.return_best_only):
        return False
    if not _alternating_optimizer_kwargs_compatible(resolved_opt):
        return False
    return _optimizer_name(resolved_opt.optimizer) in {
        "optimize_acqf",
        "optimize_acqf_mixed",
    }


class StructureAwareCandidateService(CandidateService):
    """Extend the canonical candidate service with scalable structure search."""

    def __init__(self, *, structure: Any, **kwargs: Any) -> None:
        self.structure = structure
        super().__init__(**kwargs)

    def _prepare_configs(
        self,
        owner: Any,
        acq_config: Any,
        opt_config: Any,
        values: dict[str, Any],
    ) -> tuple[Any, Any]:
        structure_ids = values.pop("structure_ids", None)
        acq_config, resolved_opt = super()._prepare_configs(
            owner,
            acq_config,
            opt_config,
            values,
        )

        model_type = str(owner.model_config.model_type).lower()
        if model_type not in _ALIGNN_MODEL_TYPES:
            if structure_ids is not None:
                raise ValueError("structure_ids is only supported for tabular ALIGNN models.")
            return acq_config, resolved_opt
        if not self.structure.enabled:
            raise ValueError(
                f"model_type={model_type!r} requires structure_col and structure_catalog."
            )
        if owner.dataset is None:
            raise RuntimeError("Call fit() before generating ALIGNN candidates.")
        try:
            structure_index = owner.dataset.feature_names.index(self.structure.column)
        except ValueError as error:
            raise RuntimeError("The fitted dataset does not contain the structure selector.") from error
        if structure_index != 0:
            raise RuntimeError("The ALIGNN structure selector must be feature index 0.")
        if resolved_opt.fixed_features_list is not None:
            raise ValueError(
                "Tabular ALIGNN derives fixed_features_list from structure_catalog and "
                "categorical process columns; use structure_ids to select a structure "
                "subset instead of supplying fixed_features_list."
            )

        structure_assignments = self.structure.fixed_features_list(
            structure_ids,
            feature_index=structure_index,
        )
        category_assignments = _process_category_fixed_features(owner)

        if _use_alternating_structure_search(
            resolved_opt,
            structure_count=len(structure_assignments),
        ):
            optimizer_kwargs = dict(resolved_opt.optimizer_kwargs)
            optimizer_kwargs.update(
                {
                    "structure_dim": structure_index,
                    "structure_values": [
                        assignment[structure_index]
                        for assignment in structure_assignments
                    ],
                    "process_fixed_features_list": category_assignments,
                }
            )
            return acq_config, replace(
                resolved_opt,
                optimizer=optimize_alignn_structure_alternating,
                optimizer_kwargs=optimizer_kwargs,
                fixed_features_list=None,
            )

        fixed_features_list = [
            {**structure_assignment, **category_assignment}
            for structure_assignment in structure_assignments
            for category_assignment in category_assignments
        ]
        return acq_config, replace(
            resolved_opt,
            fixed_features_list=fixed_features_list,
        )


__all__ = ["StructureAwareCandidateService"]
