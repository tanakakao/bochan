"""JSON-safe FastAPI schemas for CHGNet, M3GNet, and MACE residual GPs."""

from __future__ import annotations

from pydantic import model_validator

from .alignn_tabular import CrystalStructureRequest
from .tabular import TabularCandidateRequest, TabularFitModelRequest

_RESIDUAL_MODEL_TYPES = frozenset(
    {
        "chgnet_residual_gp",
        "chgnet_mixed_residual_gp",
        "chgnet_multitask_residual_gp",
        "chgnet_mixed_multitask_residual_gp",
        "chgnet_multioutput_residual_gp",
        "chgnet_mixed_multioutput_residual_gp",
        "m3gnet_residual_gp",
        "m3gnet_mixed_residual_gp",
        "m3gnet_multitask_residual_gp",
        "m3gnet_mixed_multitask_residual_gp",
        "m3gnet_multioutput_residual_gp",
        "m3gnet_mixed_multioutput_residual_gp",
        "mace_residual_gp",
        "mace_mixed_residual_gp",
        "mace_multitask_residual_gp",
        "mace_mixed_multitask_residual_gp",
        "mace_multioutput_residual_gp",
        "mace_mixed_multioutput_residual_gp",
    }
)
_MULTITASK_MODEL_TYPES = frozenset(
    model_type for model_type in _RESIDUAL_MODEL_TYPES if "multitask_residual_gp" in model_type
)
_MULTIOUTPUT_MODEL_TYPES = frozenset(
    model_type for model_type in _RESIDUAL_MODEL_TYPES if "multioutput_residual_gp" in model_type
)
_WIDE_MODEL_TYPES = _MULTITASK_MODEL_TYPES | _MULTIOUTPUT_MODEL_TYPES
_MIXED_MODEL_TYPES = frozenset(
    model_type for model_type in _RESIDUAL_MODEL_TYPES if "_mixed_" in model_type
)
_SUPPORTED_CHGNET_MODEL_NAMES = frozenset({"0.2.0", "0.3.0", "r2scan"})
_SUPPORTED_M3GNET_MODEL_NAMES = frozenset({"M3GNet-PES-MatPES-PBE-2025.2"})
_SUPPORTED_MACE_MODEL_NAMES = frozenset({"medium-mpa-0"})


def _family(model_type: str) -> str:
    return model_type.split("_", 1)[0]


def _validate_checkpoint_identifier(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("checkpoint must be a non-empty checkpoint identifier.")
    checkpoint = value.strip()
    if (
        checkpoint in {".", ".."}
        or "/" in checkpoint
        or "\\" in checkpoint
        or ":" in checkpoint
        or checkpoint.startswith("~")
    ):
        raise ValueError(
            "checkpoint must be a filename identifier, not a filesystem path. "
            "The server resolves CHGNet checkpoints under BOCHAN_CHGNET_CHECKPOINT_ROOT."
        )
    return checkpoint


class MaterialResidualTabularFitModelRequest(TabularFitModelRequest):
    """Fit scalar, correlated, or independent structure-aware residual GP models."""

    structure_col: str
    structure_catalog: dict[str, CrystalStructureRequest]

    @model_validator(mode="after")
    def validate_material_residual_contract(self):
        model_type = str(self.bo_model_config.model_type).lower()
        if model_type not in _RESIDUAL_MODEL_TYPES:
            raise ValueError(
                "Material residual FastAPI requires one of the public "
                "CHGNet/M3GNet/MACE residual GP model types."
            )
        family = _family(model_type)
        task_type = str(self.bo_model_config.task_type).lower()
        if task_type not in {"regression", "multi_objective"}:
            raise ValueError("Material residual FastAPI supports regression only.")

        targets = list(self.target_cols) if isinstance(self.target_cols, list) else [self.target_cols]
        wide = model_type in _WIDE_MODEL_TYPES
        if wide and len(targets) < 2:
            raise ValueError(f"{model_type} requires at least two continuous target columns.")
        if not wide and len(targets) != 1:
            raise ValueError(f"{model_type} requires exactly one target column.")
        if task_type == "multi_objective" and len(targets) < 2:
            raise ValueError("task_type='multi_objective' requires at least two target columns.")
        if self.multi_output_config is not None or self.bo_model_config.multi_output_config is not None:
            raise ValueError(
                "Residual output structure is derived from model_type; "
                "do not provide multi_output_config explicitly."
            )
        if self.target_categorical_cols:
            raise ValueError("Material residual FastAPI requires continuous regression targets.")

        if not self.structure_col.strip():
            raise ValueError("structure_col must be non-empty.")
        if self.structure_col not in self.input_cols:
            raise ValueError("structure_col must be included in input_cols.")
        if not self.structure_catalog:
            raise ValueError("structure_catalog must contain at least one structure.")
        if self.composition_sites:
            raise ValueError(
                "Material residual FastAPI does not yet combine composition_sites with structures."
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
        has_process_categories = bool(process_categorical)
        if (model_type in _MIXED_MODEL_TYPES) != has_process_categories:
            raise ValueError(
                "Mixed residual model types require categorical process columns, "
                "while non-mixed residual model types require continuous-only process inputs."
            )
        continuous_process_cols = [
            column
            for column in self.input_cols
            if column != self.structure_col and column not in process_categorical
        ]
        if self.bounds is not None and not isinstance(self.bounds, dict):
            raise ValueError(
                "Material residual FastAPI requires column-addressed bounds when supplied."
            )
        if continuous_process_cols and self.bounds is None:
            raise ValueError(
                "Material residual FastAPI requires bounds for every continuous process variable."
            )
        bounds = self.bounds or {}
        missing_bounds = [column for column in continuous_process_cols if column not in bounds]
        if missing_bounds:
            raise ValueError(
                "Material residual FastAPI requires bounds for every continuous process variable; "
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
                "batch_builder",
            )
            if name in model_kwargs
        )
        if forbidden:
            raise ValueError(
                "Material residual FastAPI derives encoder/structure objects server-side; "
                f"do not provide {forbidden!r} in model_kwargs."
            )
        if "encoder_training" in model_kwargs or "trainable_encoder_layers" in model_kwargs:
            raise ValueError(
                "Residual GP models keep the pretrained encoder frozen; "
                "encoder_training is not supported."
            )

        if wide:
            pretrained_output_index = model_kwargs.get("pretrained_output_index", 0)
            if isinstance(pretrained_output_index, bool) or not isinstance(pretrained_output_index, int):
                raise ValueError("pretrained_output_index must be an integer.")
            if not 0 <= pretrained_output_index < len(targets):
                raise ValueError(
                    "pretrained_output_index must select one target column: "
                    f"0 <= index < {len(targets)}."
                )
            model_kwargs["pretrained_output_index"] = pretrained_output_index
        elif "pretrained_output_index" in model_kwargs:
            raise ValueError(
                "pretrained_output_index is only valid for multitask or multioutput residual models."
            )

        if family == "chgnet":
            model_name = str(model_kwargs.get("model_name", "0.3.0"))
            if model_name not in _SUPPORTED_CHGNET_MODEL_NAMES:
                raise ValueError(
                    f"model_name must be one of {sorted(_SUPPORTED_CHGNET_MODEL_NAMES)!r}."
                )
            model_kwargs["model_name"] = model_name
            if model_kwargs.get("checkpoint") is not None:
                model_kwargs["checkpoint"] = _validate_checkpoint_identifier(
                    model_kwargs["checkpoint"]
                )
        elif family == "m3gnet":
            model_name = str(
                model_kwargs.get("model_name", "M3GNet-PES-MatPES-PBE-2025.2")
            )
            if model_name not in _SUPPORTED_M3GNET_MODEL_NAMES:
                raise ValueError(
                    f"model_name must be one of {sorted(_SUPPORTED_M3GNET_MODEL_NAMES)!r}."
                )
            model_kwargs["model_name"] = model_name
        else:
            model_name = str(model_kwargs.get("model_name", "medium-mpa-0"))
            if model_name not in _SUPPORTED_MACE_MODEL_NAMES:
                raise ValueError(
                    f"model_name must be one of {sorted(_SUPPORTED_MACE_MODEL_NAMES)!r}."
                )
            model_kwargs["model_name"] = model_name
            num_layers = model_kwargs.get("num_layers", -1)
            if isinstance(num_layers, bool) or not isinstance(num_layers, int):
                raise ValueError("num_layers must be -1 or a positive integer.")
            if num_layers != -1 and num_layers <= 0:
                raise ValueError("num_layers must be -1 or a positive integer.")
            model_kwargs["num_layers"] = num_layers
            pooling = str(model_kwargs.get("pooling", "mean")).lower()
            if pooling not in {"mean", "sum"}:
                raise ValueError("pooling must be 'mean' or 'sum'.")
            model_kwargs["pooling"] = pooling
            head = model_kwargs.get("head")
            if head is not None and (not isinstance(head, str) or not head.strip()):
                raise ValueError("head must be a non-empty string when provided.")

        self.bo_model_config.model_kwargs = model_kwargs
        return self


class MaterialResidualTabularCandidateRequest(TabularCandidateRequest):
    """Generate candidates across all or a selected subset of known structures."""

    structure_ids: list[str] | str | None = None


__all__ = [
    "MaterialResidualTabularCandidateRequest",
    "MaterialResidualTabularFitModelRequest",
]
