"""Structure-aware candidate generation for tabular ALIGNN models."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from ..optimizer.candidates import CandidateService

_ALIGNN_MODEL_TYPES = frozenset({"alignn_gp", "alignn_dkl"})


def _process_category_fixed_features(owner: Any) -> list[dict[int, float]]:
    """Return observed joint process-category assignments for mixed ALIGNN."""

    if owner.bo.bundle is None or owner.dataset is None:
        return [{}]
    cat_dims = [int(index) for index in (owner.bo.bundle.cat_dims or [])]
    if not cat_dims:
        return [{}]
    if 0 in cat_dims:
        raise RuntimeError(
            "ALIGNN model cat_dims must contain process categories only; "
            "structure feature 0 is enumerated separately."
        )

    import torch

    values = owner.dataset.X[:, cat_dims]
    unique_rows = torch.unique(values, dim=0)
    if unique_rows.numel() == 0:
        raise RuntimeError("No categorical process assignments are available for optimization.")
    return [
        {dim: float(value) for dim, value in zip(cat_dims, row, strict=True)}
        for row in unique_rows.detach().cpu().tolist()
    ]


class StructureAwareCandidateService(CandidateService):
    """Extend the canonical candidate service with structure enumeration."""

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
