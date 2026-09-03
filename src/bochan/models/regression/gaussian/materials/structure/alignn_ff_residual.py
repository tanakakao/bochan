"""ALIGNN-FF deterministic baselines and residual Gaussian processes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from importlib import import_module
from typing import Any, Literal

import torch
from gpytorch.likelihoods import Likelihood
from torch import Tensor, nn

from bochan.composition import ALIGNNEncoder, MaterialProcessFusion
from bochan.composition.encoders.alignn import Checkpoint
from bochan.models.regression.gaussian.deep.alignn import ALIGNNGPModel, _resolve_material_encoder
from bochan.models.regression.gaussian.deep.alignn_multitask import ALIGNNMultiTaskGPModel
from bochan.models.regression.gaussian.deep.deepkernel import InputTransformArg, OutcomeTransformArg
from bochan.models.regression.gaussian.materials.common.baseline import MaterialBaselineSpec, MaterialPropertyContract
from bochan.models.regression.gaussian.materials.common.residual import (
    DirectMaterialPredictor,
    ResidualMaterialGPModel,
    compute_material_residual_targets,
)
from bochan.models.regression.gaussian.materials.common.tensor_target import TensorTargetLayout
from bochan.structure.adapter import StructureAdapter

_ALIGNNFFKind = Literal["energy", "force", "stress"]
_DEFAULT_MODEL_NAME = "alignnff_wt10"


def _validate_structure_bank(structures: Sequence[Any]) -> tuple[Any, ...]:
    if isinstance(structures, (str, bytes)) or not isinstance(structures, Sequence):
        raise TypeError("structures must be a non-empty sequence.")
    resolved = tuple(structures)
    if not resolved:
        raise ValueError("structures must contain at least one structure.")
    return resolved


def _load_alignn_ff_calculator(model_name: str) -> Any:
    """Load the current ALIGNN-FF ASE calculator, preferring the unified API."""
    try:
        module = import_module("alignn.ff.unified_calculator")
        config_class = getattr(module, "AlignnUnifiedConfig", None)
        calculator_class = getattr(module, "AlignnUnifiedCalculator", None)
        if isinstance(config_class, type) and isinstance(calculator_class, type):
            try:
                config = config_class(energy=True, forces=True, stress=True, model_name=model_name)
            except TypeError:
                config = config_class(energy=True, forces=True, stress=True)
            return calculator_class(config)
    except ImportError:
        pass

    try:
        module = import_module("alignn.ff.ff")
    except ImportError as error:
        raise ImportError(
            "ALIGNN-FF support requires alignn>=2026.8.11. Install bochan[materials]."
        ) from error
    calculator_class = getattr(module, "AlignnAtomwiseCalculator", None)
    if not isinstance(calculator_class, type):
        raise RuntimeError("The installed ALIGNN package exposes neither unified nor legacy ASE calculator.")
    default_path = getattr(module, "default_path", None)
    path = default_path() if callable(default_path) else model_name
    return calculator_class(path=path)


def _structure_indices(X: Tensor, *, num_structures: int) -> tuple[Tensor, torch.Size]:
    if not torch.is_tensor(X):
        raise TypeError("X must be a Tensor.")
    if X.ndim < 2 or X.shape[-1] < 1:
        raise ValueError("X must have shape [..., q, 1 + process_dim].")
    flat = X.reshape(-1, X.shape[-1])
    raw = flat[:, 0]
    if not torch.isfinite(raw).all():
        raise ValueError("Structure indices must be finite.")
    rounded = raw.round()
    if not torch.equal(raw, rounded):
        raise ValueError("Structure indices must be integer-valued.")
    indices = rounded.to(dtype=torch.long)
    if indices.numel() and (int(indices.min()) < 0 or int(indices.max()) >= num_structures):
        raise ValueError("Structure index is outside the configured structure bank.")
    return indices, X.shape[:-1]


def _infer_num_atoms(structure: Any) -> int | None:
    if isinstance(structure, Mapping):
        for key in ("elements", "atomic_numbers", "species"):
            value = structure.get(key)
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                return len(value)
    try:
        value = len(structure)
    except (TypeError, AttributeError):
        return None
    return int(value) if value > 0 else None


def _resolve_num_atoms(structures: Sequence[Any], num_atoms: int | None) -> int:
    if num_atoms is not None:
        if isinstance(num_atoms, bool) or not isinstance(num_atoms, int) or num_atoms <= 0:
            raise ValueError("num_atoms must be a positive integer.")
        return num_atoms
    inferred = [_infer_num_atoms(structure) for structure in structures]
    if any(value is None for value in inferred):
        raise ValueError("Could not infer fixed atom count; pass num_atoms explicitly.")
    expected = int(inferred[0])
    if any(int(value) != expected for value in inferred if value is not None):
        raise ValueError("ALIGNN-FF force residual requires fixed topology across structures.")
    return expected


class _ALIGNNFFDirectPredictor(DirectMaterialPredictor):
    def __init__(
        self,
        structures: Sequence[Any],
        *,
        kind: _ALIGNNFFKind,
        model_name: str = _DEFAULT_MODEL_NAME,
        calculator: Any | None = None,
        adapter: StructureAdapter | None = None,
        num_atoms: int | None = None,
    ) -> None:
        super().__init__()
        self.structures = _validate_structure_bank(structures)
        self.kind = kind
        self.model_name = model_name
        self.adapter = StructureAdapter() if adapter is None else adapter
        if not isinstance(self.adapter, StructureAdapter):
            raise TypeError("adapter must be a StructureAdapter.")
        self.calculator = _load_alignn_ff_calculator(model_name) if calculator is None else calculator
        if kind == "force":
            self.layout = TensorTargetLayout.force(_resolve_num_atoms(self.structures, num_atoms))
        elif kind == "stress":
            self.layout = TensorTargetLayout.stress()
        else:
            self.layout = None
        self.register_buffer("_cached_baseline", torch.empty(0), persistent=False)

    @property
    def output_dim(self) -> int:
        return 1 if self.layout is None else self.layout.output_dim

    def _predict_one(self, structure: Any) -> Tensor:
        atoms = self.adapter.to_ase(structure).copy()
        atoms.calc = self.calculator
        if self.kind == "energy":
            value = torch.tensor([[float(atoms.get_potential_energy())]], dtype=torch.get_default_dtype())
        elif self.kind == "force":
            raw = torch.as_tensor(atoms.get_forces(), dtype=torch.get_default_dtype())
            if tuple(raw.shape) != self.layout.tensor_shape:
                raise ValueError(f"ALIGNN-FF forces must have shape {self.layout.tensor_shape}; got {tuple(raw.shape)}.")
            value = self.layout.flatten(raw).reshape(1, self.output_dim)
        else:
            raw = torch.as_tensor(atoms.get_stress(voigt=False), dtype=torch.get_default_dtype())
            if tuple(raw.shape) != (3, 3):
                raise ValueError(f"ALIGNN-FF stress must have shape (3, 3); got {tuple(raw.shape)}.")
            value = self.layout.flatten(raw).reshape(1, self.output_dim)
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"ALIGNN-FF produced non-finite {self.kind} prediction.")
        return value

    def _baseline_bank(self) -> Tensor:
        if self._cached_baseline.numel() == 0:
            self._cached_baseline = torch.cat([self._predict_one(s) for s in self.structures], dim=0).detach()
        return self._cached_baseline

    def clear_cache(self) -> None:
        self._cached_baseline = self._cached_baseline.new_empty(0)

    def forward(self, X: Tensor) -> Tensor:
        indices, leading = _structure_indices(X, num_structures=len(self.structures))
        bank = self._baseline_bank()
        selected = bank[indices.to(device=bank.device)]
        return selected.to(device=X.device, dtype=X.dtype).reshape(*leading, self.output_dim)


class ALIGNNFFDirectEnergyPredictor(_ALIGNNFFDirectPredictor):
    def __init__(self, structures: Sequence[Any], **kwargs: Any) -> None:
        super().__init__(structures, kind="energy", **kwargs)


class ALIGNNFFDirectForcePredictor(_ALIGNNFFDirectPredictor):
    def __init__(self, structures: Sequence[Any], *, num_atoms: int | None = None, **kwargs: Any) -> None:
        super().__init__(structures, kind="force", num_atoms=num_atoms, **kwargs)
        self.num_atoms = self.layout.tensor_shape[0]


class ALIGNNFFDirectStressPredictor(_ALIGNNFFDirectPredictor):
    def __init__(self, structures: Sequence[Any], **kwargs: Any) -> None:
        super().__init__(structures, kind="stress", **kwargs)


def _baseline_spec(kind: _ALIGNNFFKind, model_name: str, contract: MaterialPropertyContract | None) -> MaterialBaselineSpec | None:
    if contract is None:
        return None
    if contract.quantity.casefold() != kind:
        raise ValueError(f"ALIGNN-FF {kind} residual requires target_contract.quantity={kind!r}.")
    return MaterialBaselineSpec(family="alignn-ff", property=contract, model_name=model_name)


class ALIGNNFFEnergyResidualGPModel(ResidualMaterialGPModel):
    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor | None = None,
        *,
        structures: Sequence[Any],
        structure_graphs: Sequence[Any],
        encoder: ALIGNNEncoder | nn.Module | None = None,
        checkpoint: Checkpoint | None = None,
        encoder_output_dim: int | None = None,
        encoder_config: object | None = None,
        latent_dim: int = 32,
        fusion: Literal["concat"] | MaterialProcessFusion = "concat",
        projection: nn.Module | None = None,
        strict_checkpoint: bool = True,
        likelihood: Likelihood | None = None,
        input_transform: InputTransformArg = "DEFAULT",
        outcome_transform: OutcomeTransformArg = "DEFAULT",
        model_name: str = _DEFAULT_MODEL_NAME,
        calculator: Any | None = None,
        adapter: StructureAdapter | None = None,
        target_contract: MaterialPropertyContract | None = None,
    ) -> None:
        if len(structures) != len(structure_graphs):
            raise ValueError("structures and structure_graphs must have the same length and order.")
        material_encoder = _resolve_material_encoder(
            encoder, checkpoint, output_dim=encoder_output_dim, config=encoder_config, strict_checkpoint=strict_checkpoint
        )
        predictor = ALIGNNFFDirectEnergyPredictor(structures, model_name=model_name, calculator=calculator, adapter=adapter)
        spec = _baseline_spec("energy", model_name, target_contract)
        residual_y = compute_material_residual_targets(train_X, train_Y, predictor, baseline_spec=spec, target_contract=target_contract)
        residual_model = ALIGNNGPModel(
            train_X, residual_y, train_Yvar,
            structure_graphs=structure_graphs, encoder=material_encoder, encoder_output_dim=encoder_output_dim,
            latent_dim=latent_dim, fusion=fusion, projection=projection, likelihood=likelihood,
            input_transform=input_transform, outcome_transform=outcome_transform,
        )
        super().__init__(predictor=predictor, residual_model=residual_model, baseline_spec=spec)
        self.material_encoder = material_encoder
        self.structures = tuple(structures)
        self.structure_graphs = tuple(structure_graphs)


class _ALIGNNFFTensorResidualGPModel(ResidualMaterialGPModel):
    kind: Literal["force", "stress"]

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor | None,
        *,
        structures: Sequence[Any],
        structure_graphs: Sequence[Any],
        kind: Literal["force", "stress"],
        num_atoms: int | None = None,
        encoder: ALIGNNEncoder | nn.Module | None = None,
        checkpoint: Checkpoint | None = None,
        encoder_output_dim: int | None = None,
        encoder_config: object | None = None,
        latent_dim: int = 32,
        fusion: Literal["concat"] | MaterialProcessFusion = "concat",
        projection: nn.Module | None = None,
        strict_checkpoint: bool = True,
        likelihood: Likelihood | None = None,
        input_transform: InputTransformArg = "DEFAULT",
        outcome_transform: OutcomeTransformArg = "DEFAULT",
        model_name: str = _DEFAULT_MODEL_NAME,
        calculator: Any | None = None,
        adapter: StructureAdapter | None = None,
        target_contract: MaterialPropertyContract | None = None,
    ) -> None:
        if len(structures) != len(structure_graphs):
            raise ValueError("structures and structure_graphs must have the same length and order.")
        material_encoder = _resolve_material_encoder(
            encoder, checkpoint, output_dim=encoder_output_dim, config=encoder_config, strict_checkpoint=strict_checkpoint
        )
        predictor: _ALIGNNFFDirectPredictor
        if kind == "force":
            predictor = ALIGNNFFDirectForcePredictor(structures, num_atoms=num_atoms, model_name=model_name, calculator=calculator, adapter=adapter)
        else:
            predictor = ALIGNNFFDirectStressPredictor(structures, model_name=model_name, calculator=calculator, adapter=adapter)
        flat_y = predictor.layout.flatten(train_Y, n=train_X.shape[0])
        flat_yvar = None if train_Yvar is None else predictor.layout.flatten(train_Yvar, n=train_X.shape[0])
        spec = _baseline_spec(kind, model_name, target_contract)
        residual_y = compute_material_residual_targets(train_X, flat_y, predictor, baseline_spec=spec, target_contract=target_contract)
        residual_model = ALIGNNMultiTaskGPModel(
            train_X, residual_y, flat_yvar,
            structure_graphs=structure_graphs, encoder=material_encoder, encoder_output_dim=encoder_output_dim,
            latent_dim=latent_dim, fusion=fusion, projection=projection, likelihood=likelihood,
            input_transform=input_transform, outcome_transform=outcome_transform,
        )
        super().__init__(predictor=predictor, residual_model=residual_model, baseline_spec=spec)
        self.layout = predictor.layout
        self.material_encoder = material_encoder
        self.structures = tuple(structures)
        self.structure_graphs = tuple(structure_graphs)
        if kind == "force":
            self.num_atoms = predictor.layout.tensor_shape[0]

    def unflatten(self, values: Tensor) -> Tensor:
        return self.layout.unflatten(values)


class ALIGNNFFForceResidualGPModel(_ALIGNNFFTensorResidualGPModel):
    def __init__(self, train_X: Tensor, train_Y: Tensor, train_Yvar: Tensor | None = None, **kwargs: Any) -> None:
        super().__init__(train_X, train_Y, train_Yvar, kind="force", **kwargs)


class ALIGNNFFStressResidualGPModel(_ALIGNNFFTensorResidualGPModel):
    def __init__(self, train_X: Tensor, train_Y: Tensor, train_Yvar: Tensor | None = None, **kwargs: Any) -> None:
        super().__init__(train_X, train_Y, train_Yvar, kind="stress", **kwargs)


__all__ = [
    "ALIGNNFFDirectEnergyPredictor",
    "ALIGNNFFDirectForcePredictor",
    "ALIGNNFFDirectStressPredictor",
    "ALIGNNFFEnergyResidualGPModel",
    "ALIGNNFFForceResidualGPModel",
    "ALIGNNFFStressResidualGPModel",
]
