"""JSON-safe FastAPI schemas for pure-PyTorch tabular ALIGNN models."""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import Field, model_validator

from .requests import APIRequest
from .tabular import TabularCandidateRequest, TabularFitModelRequest

_ALIGNN_MODEL_TYPES = frozenset({"alignn_gp", "alignn_dkl"})
_PURE_MODEL_NAME = "alignn_atomwise_pure"


class CrystalStructureRequest(APIRequest):
    """One crystal structure supplied inline over HTTP.

    ``format='mapping'`` uses the same lattice/coordinate contract as
    :class:`bochan.structure.StructureAdapter`. ``cif`` and ``poscar`` accept
    file contents, never a server-side path supplied by the client.
    """

    format: Literal["mapping", "cif", "poscar"] = "mapping"
    lattice_mat: list[list[float]] | None = None
    coords: list[list[float]] | None = None
    elements: list[str] | None = None
    cartesian: bool = False
    content: str | None = None

    @model_validator(mode="after")
    def validate_structure_payload(self):
        if self.format in {"cif", "poscar"}:
            if not isinstance(self.content, str) or not self.content.strip():
                raise ValueError(f"format={self.format!r} requires non-empty content.")
            if any(
                value is not None
                for value in (self.lattice_mat, self.coords, self.elements)
            ):
                raise ValueError(
                    "CIF/POSCAR payloads use content only; do not also pass "
                    "lattice_mat, coords, or elements."
                )
            return self

        if self.content is not None:
            raise ValueError("format='mapping' does not accept content.")
        if self.lattice_mat is None or self.coords is None or self.elements is None:
            raise ValueError(
                "format='mapping' requires lattice_mat, coords, and elements."
            )
        if len(self.lattice_mat) != 3 or any(len(row) != 3 for row in self.lattice_mat):
            raise ValueError("lattice_mat must have shape [3, 3].")
        if not self.coords or any(len(row) != 3 for row in self.coords):
            raise ValueError("coords must have shape [n_atoms, 3].")
        if len(self.elements) != len(self.coords):
            raise ValueError("elements and coords must contain the same number of atoms.")
        if any(not str(element).strip() for element in self.elements):
            raise ValueError("elements must contain non-empty element symbols.")
        numbers = [value for row in self.lattice_mat for value in row]
        numbers.extend(value for row in self.coords for value in row)
        if not all(math.isfinite(float(value)) for value in numbers):
            raise ValueError("lattice_mat and coords must contain only finite values.")
        return self

    def as_mapping(self) -> dict[str, Any]:
        """Return the canonical in-memory structure mapping."""

        if self.format != "mapping":
            raise ValueError("Only format='mapping' can be converted directly to a mapping.")
        return {
            "lattice_mat": self.lattice_mat,
            "coords": self.coords,
            "elements": self.elements,
            "cartesian": self.cartesian,
        }


class ALIGNNGraphConfigRequest(APIRequest):
    """JSON-safe pure-PyTorch ALIGNN graph settings."""

    neighbor_strategy: Literal["pure_torch"] = "pure_torch"
    cutoff: float = Field(default=8.0, gt=0.0)
    max_neighbors: int | None = Field(default=12, ge=1)
    atom_features: str = "cgcnn"
    compute_line_graph: bool = True
    dtype: Literal["float16", "float32", "float64", "bfloat"] = "float32"
    three_body_cutoff: float | None = Field(default=3.5, gt=0.0)

    @model_validator(mode="after")
    def validate_graph_config(self):
        if not self.atom_features.strip():
            raise ValueError("atom_features must be non-empty.")
        if not self.compute_line_graph:
            raise ValueError("FastAPI ALIGNN currently requires compute_line_graph=True.")
        if self.three_body_cutoff is not None and self.three_body_cutoff > self.cutoff:
            raise ValueError("three_body_cutoff must not exceed cutoff.")
        return self


class ALIGNNTabularFitModelRequest(TabularFitModelRequest):
    """Fit pure-PyTorch ALIGNN-GP / ALIGNN-DKL from tabular data and structures."""

    structure_col: str
    structure_catalog: dict[str, CrystalStructureRequest]
    structure_graph_config: ALIGNNGraphConfigRequest = Field(
        default_factory=ALIGNNGraphConfigRequest
    )

    @model_validator(mode="after")
    def validate_alignn_contract(self):
        model_type = str(self.bo_model_config.model_type).lower()
        if model_type not in _ALIGNN_MODEL_TYPES:
            raise ValueError(
                "ALIGNN tabular FastAPI requires model_type='alignn_gp' or 'alignn_dkl'."
            )
        if str(self.bo_model_config.task_type).lower() != "regression":
            raise ValueError("ALIGNN tabular FastAPI currently supports regression only.")
        targets = list(self.target_cols) if isinstance(self.target_cols, list) else [self.target_cols]
        if len(targets) != 1:
            raise ValueError("ALIGNN tabular FastAPI currently requires one target column.")
        if (
            self.multi_output_config is not None
            or self.bo_model_config.multi_output_config is not None
        ):
            raise ValueError("ALIGNN tabular FastAPI does not support multi_output_config yet.")
        if self.target_categorical_cols:
            raise ValueError("ALIGNN tabular FastAPI requires a continuous regression target.")
        if not self.structure_col.strip():
            raise ValueError("structure_col must be non-empty.")
        if self.structure_col not in self.input_cols:
            raise ValueError("structure_col must be included in input_cols.")
        if not self.structure_catalog:
            raise ValueError("structure_catalog must contain at least one structure.")
        if self.composition_sites:
            raise ValueError(
                "ALIGNN FastAPI does not yet combine composition_sites with crystal structures."
            )
        other_categorical = [
            column for column in self.categorical_cols if column != self.structure_col
        ]
        if other_categorical:
            raise ValueError(
                "ALIGNN FastAPI Phase 3 supports continuous process variables only; "
                f"categorical process columns were configured: {other_categorical!r}."
            )
        if not isinstance(self.bounds, dict):
            raise ValueError(
                "ALIGNN FastAPI requires column-addressed bounds for process variables."
            )
        process_cols = [column for column in self.input_cols if column != self.structure_col]
        missing_bounds = [column for column in process_cols if column not in self.bounds]
        if missing_bounds:
            raise ValueError(
                "ALIGNN FastAPI requires bounds for every continuous process variable; "
                f"missing bounds for {missing_bounds!r}."
            )

        known_ids = set(self.structure_catalog)
        if isinstance(self.data, list):
            observed_ids = {
                str(row[self.structure_col])
                for row in self.data
                if self.structure_col in row
            }
            missing_rows = [
                index
                for index, row in enumerate(self.data)
                if self.structure_col not in row
            ]
            if missing_rows:
                raise ValueError(f"structure_col is missing from data rows {missing_rows!r}.")
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
        if "structure_graphs" in model_kwargs:
            raise ValueError(
                "structure_graphs is derived from structure_catalog and cannot be supplied over HTTP."
            )
        if "trainable_encoder_layers" in model_kwargs:
            raise ValueError(
                "Use model_config.model_kwargs.encoder_training='partial' or 'full' "
                "instead of trainable_encoder_layers over FastAPI."
            )

        encoder_config = model_kwargs.get("encoder_config")
        if encoder_config is not None:
            if not isinstance(encoder_config, dict):
                raise ValueError("encoder_config must be a JSON object when provided.")
            encoder_name = encoder_config.get("name", _PURE_MODEL_NAME)
            if encoder_name != _PURE_MODEL_NAME:
                raise ValueError(
                    "FastAPI ALIGNN uses the pure-PyTorch encoder; "
                    f"encoder_config.name must be {_PURE_MODEL_NAME!r}."
                )
            encoder_config = dict(encoder_config)
            encoder_config["name"] = _PURE_MODEL_NAME
            model_kwargs["encoder_config"] = encoder_config

        checkpoint = model_kwargs.get("checkpoint")
        if checkpoint is not None:
            if not isinstance(checkpoint, str) or not checkpoint.strip():
                raise ValueError("checkpoint must be a non-empty checkpoint identifier.")
            checkpoint = checkpoint.strip()
            if (
                checkpoint in {".", ".."}
                or "/" in checkpoint
                or "\\" in checkpoint
                or ":" in checkpoint
                or checkpoint.startswith("~")
            ):
                raise ValueError(
                    "checkpoint must be a filename identifier, not a filesystem path. "
                    "The server resolves it under BOCHAN_ALIGNN_CHECKPOINT_ROOT."
                )
            model_kwargs["checkpoint"] = checkpoint
        if model_type == "alignn_gp" and "encoder_training" in model_kwargs:
            raise ValueError("alignn_gp always freezes the ALIGNN encoder.")
        if model_type == "alignn_dkl" and "encoder_training" in model_kwargs:
            training = str(model_kwargs["encoder_training"]).lower()
            if training not in {"partial", "full"}:
                raise ValueError("encoder_training must be 'partial' or 'full'.")
            model_kwargs["encoder_training"] = training
        self.bo_model_config.model_kwargs = model_kwargs
        return self


class ALIGNNTabularCandidateRequest(TabularCandidateRequest):
    """Generate candidates across all or a selected subset of known structures."""

    structure_ids: list[str] | str | None = None


__all__ = [
    "ALIGNNGraphConfigRequest",
    "ALIGNNTabularCandidateRequest",
    "ALIGNNTabularFitModelRequest",
    "CrystalStructureRequest",
]
