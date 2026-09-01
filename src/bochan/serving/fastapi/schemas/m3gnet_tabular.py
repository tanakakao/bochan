"""JSON-safe FastAPI schemas for structure-aware M3GNet tabular models."""

from __future__ import annotations

from pydantic import model_validator

from .alignn_tabular import CrystalStructureRequest
from .tabular import TabularCandidateRequest, TabularFitModelRequest

_M3GNET_MODEL_TYPES = frozenset(
    {"m3gnet_gp", "m3gnet_dkl", "m3gnet_multitask", "m3gnet_multitask_dkl"}
)
_M3GNET_MULTITASK_MODEL_TYPES = frozenset({"m3gnet_multitask", "m3gnet_multitask_dkl"})
_M3GNET_FROZEN_MODEL_TYPES = frozenset({"m3gnet_gp", "m3gnet_multitask"})
_M3GNET_DKL_MODEL_TYPES = frozenset({"m3gnet_dkl", "m3gnet_multitask_dkl"})
_SUPPORTED_MODEL_NAMES = frozenset({"M3GNet-PES-MatPES-PBE-2025.2"})


class M3GNetTabularFitModelRequest(TabularFitModelRequest):
    """Fit M3GNet GP/DKL, including independent and correlated multi-output variants."""

    structure_col: str
    structure_catalog: dict[str, CrystalStructureRequest]

    @model_validator(mode="after")
    def validate_m3gnet_contract(self):
        model_type = str(self.bo_model_config.model_type).lower()
        if model_type not in _M3GNET_MODEL_TYPES:
            raise ValueError(
                "M3GNet tabular FastAPI requires model_type='m3gnet_gp', 'm3gnet_dkl', "
                "'m3gnet_multitask', or 'm3gnet_multitask_dkl'."
            )
        task_type = str(self.bo_model_config.task_type).lower()
        if task_type not in {"regression", "multi_objective"}:
            raise ValueError(
                "M3GNet tabular FastAPI supports regression or multi_objective regression only."
            )
        targets = list(self.target_cols) if isinstance(self.target_cols, list) else [self.target_cols]
        if not targets:
            raise ValueError("M3GNet tabular FastAPI requires at least one target column.")
        if model_type in _M3GNET_MULTITASK_MODEL_TYPES and len(targets) < 2:
            fallback = "m3gnet_dkl" if model_type.endswith("_dkl") else "m3gnet_gp"
            raise ValueError(
                f"{model_type} requires at least two continuous target columns. "
                f"Use model_type={fallback!r} for a single target."
            )
        if task_type == "multi_objective" and len(targets) < 2:
            raise ValueError(
                "M3GNet task_type='multi_objective' requires at least two target columns."
            )
        if self.multi_output_config is not None or self.bo_model_config.multi_output_config is not None:
            if model_type in _M3GNET_MULTITASK_MODEL_TYPES:
                raise ValueError(
                    "Correlated M3GNet multitask models keep wide targets in one model; "
                    "do not provide multi_output_config."
                )
            raise ValueError(
                "M3GNet FastAPI derives independent multi-output models automatically "
                "from target_cols; do not provide multi_output_config explicitly."
            )
        if self.target_categorical_cols:
            raise ValueError("M3GNet tabular FastAPI requires continuous regression targets.")
        if not self.structure_col.strip():
            raise ValueError("structure_col must be non-empty.")
        if self.structure_col not in self.input_cols:
            raise ValueError("structure_col must be included in input_cols.")
        if not self.structure_catalog:
            raise ValueError("structure_catalog must contain at least one structure.")
        if self.composition_sites:
            raise ValueError(
                "M3GNet FastAPI does not yet combine composition_sites with crystal structures."
            )

        unknown_categorical = [
            column for column in self.categorical_cols if column not in self.input_cols
        ]
        if unknown_categorical:
            raise ValueError(
                "categorical_cols must be included in input_cols; "
                f"unknown columns: {unknown_categorical!r}."
            )
        process_categorical = {
            column for column in self.categorical_cols if column != self.structure_col
        }
        continuous_process_cols = [
            column
            for column in self.input_cols
            if column != self.structure_col and column not in process_categorical
        ]
        if self.bounds is not None and not isinstance(self.bounds, dict):
            raise ValueError(
                "M3GNet FastAPI requires column-addressed bounds when bounds are supplied."
            )
        if continuous_process_cols and self.bounds is None:
            raise ValueError(
                "M3GNet FastAPI requires column-addressed bounds for continuous process variables."
            )
        bounds = self.bounds or {}
        missing_bounds = [column for column in continuous_process_cols if column not in bounds]
        if missing_bounds:
            raise ValueError(
                "M3GNet FastAPI requires bounds for every continuous process variable; "
                f"missing bounds for {missing_bounds!r}."
            )

        known_ids = set(self.structure_catalog)
        if isinstance(self.data, list):
            missing_rows = [
                index for index, row in enumerate(self.data) if self.structure_col not in row
            ]
            if missing_rows:
                raise ValueError(f"structure_col is missing from data rows {missing_rows!r}.")
            observed_ids = {str(row[self.structure_col]) for row in self.data}
        else:
            if self.structure_col not in self.data:
                raise ValueError("structure_col must be present in data.")
            observed_ids = {str(value) for value in self.data[self.structure_col]}
        unknown_ids = sorted(observed_ids - known_ids)
        if unknown_ids:
            raise ValueError(
                "Every structure ID in data must exist in structure_catalog; "
                f"unknown IDs: {unknown_ids!r}."
            )

        model_kwargs = dict(self.bo_model_config.model_kwargs)
        forbidden = sorted(
            name
            for name in (
                "encoder",
                "adapter",
                "structures",
                "structure_graphs",
                "graph_converter",
            )
            if name in model_kwargs
        )
        if forbidden:
            raise ValueError(
                "M3GNet FastAPI derives encoder/structure objects server-side; "
                f"do not provide {forbidden!r} in model_kwargs."
            )
        if "trainable_encoder_layers" in model_kwargs:
            raise ValueError(
                "Use model_config.model_kwargs.encoder_training='partial' or 'full' "
                "instead of trainable_encoder_layers over FastAPI."
            )

        model_name = str(model_kwargs.get("model_name", "M3GNet-PES-MatPES-PBE-2025.2"))
        if model_name not in _SUPPORTED_MODEL_NAMES:
            raise ValueError(
                f"model_name must be one of {sorted(_SUPPORTED_MODEL_NAMES)!r}."
            )
        model_kwargs["model_name"] = model_name

        if model_type in _M3GNET_FROZEN_MODEL_TYPES and "encoder_training" in model_kwargs:
            raise ValueError(f"{model_type} always freezes the M3GNet encoder.")
        if model_type in _M3GNET_DKL_MODEL_TYPES and "encoder_training" in model_kwargs:
            training = str(model_kwargs["encoder_training"]).lower()
            if training not in {"partial", "full"}:
                raise ValueError("encoder_training must be 'partial' or 'full'.")
            model_kwargs["encoder_training"] = training
        self.bo_model_config.model_kwargs = model_kwargs
        return self


class M3GNetTabularCandidateRequest(TabularCandidateRequest):
    """Generate candidates across all or a selected subset of known structures."""

    structure_ids: list[str] | str | None = None


__all__ = [
    "M3GNetTabularCandidateRequest",
    "M3GNetTabularFitModelRequest",
]
