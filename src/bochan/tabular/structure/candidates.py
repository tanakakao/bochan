"""Structure-aware candidate generation for tabular ALIGNN models."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from ..optimizer.candidates import CandidateService

_ALIGNN_MODEL_TYPES = frozenset({"alignn_gp", "alignn_dkl"})


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
                "Tabular ALIGNN derives fixed_features_list from structure_catalog; "
                "use structure_ids to select a subset instead of supplying fixed_features_list."
            )

        return acq_config, replace(
            resolved_opt,
            fixed_features_list=self.structure.fixed_features_list(
                structure_ids,
                feature_index=structure_index,
            ),
        )


__all__ = ["StructureAwareCandidateService"]
