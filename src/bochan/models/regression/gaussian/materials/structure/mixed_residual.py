"""Mixed-input residual Gaussian processes for pretrained MLIP models.

Categorical process variables are an input-space capability orthogonal to the
existing ``residual_gp`` model mode.  Energy residuals use the backend's scalar
mixed GP, while force and stress residuals use the backend's correlated mixed
multi-output GP so cross-component covariance is retained.
"""

from __future__ import annotations

from collections.abc import Sequence
from importlib import import_module
from typing import Any

from torch import Tensor

from bochan.models.regression.gaussian.materials.common.residual import (
    DirectMaterialPredictor,
    ResidualMaterialGPModel,
    compute_material_residual_targets,
)

from .factory import MaterialMLIPBackend


def _load(module_name: str) -> Any:
    return import_module(module_name, package=__package__)


def _pop(kwargs: dict[str, Any], name: str, default: Any) -> Any:
    return kwargs.pop(name, default)


def _require_cat_dims(cat_dims: Sequence[int]) -> tuple[int, ...]:
    if isinstance(cat_dims, (str, bytes)) or not isinstance(cat_dims, Sequence):
        raise TypeError("cat_dims must be a non-empty sequence of process-column indices.")
    resolved = tuple(cat_dims)
    if not resolved:
        raise ValueError(
            "mixed residual GP requires at least one categorical process dimension."
        )
    if any(isinstance(index, bool) or not isinstance(index, int) for index in resolved):
        raise TypeError("cat_dims must contain integer column indices.")
    if 0 in resolved:
        raise ValueError(
            "cat_dims must not include column 0; the structure-index column is "
            "handled by the material encoder."
        )
    return resolved


def _reject_unused(kwargs: dict[str, Any], *, backend: str, quantity: str) -> None:
    if kwargs:
        names = ", ".join(sorted(kwargs))
        raise TypeError(
            f"Unsupported mixed residual arguments for {backend}/{quantity}: {names}."
        )


class MixedScalarResidualMaterialGPModel(ResidualMaterialGPModel):
    """Scalar mixed residual wrapper carrying backend/input metadata."""

    def __init__(
        self,
        *,
        predictor: DirectMaterialPredictor,
        residual_model: Any,
        backend: str,
        cat_dims: Sequence[int],
        baseline_spec: Any = None,
        material_encoder: Any = None,
    ) -> None:
        super().__init__(
            predictor=predictor,
            residual_model=residual_model,
            baseline_spec=baseline_spec,
        )
        self.backend = backend
        self.cat_dims = tuple(cat_dims)
        self.material_encoder = material_encoder
        self.structures = tuple(getattr(predictor, "structures", ()))


class MixedTensorResidualMaterialGPModel(ResidualMaterialGPModel):
    """Mixed force/stress residual wrapper retaining tensor layout helpers."""

    def __init__(
        self,
        *,
        predictor: DirectMaterialPredictor,
        residual_model: Any,
        backend: str,
        cat_dims: Sequence[int],
        baseline_spec: Any = None,
        material_encoder: Any = None,
    ) -> None:
        super().__init__(
            predictor=predictor,
            residual_model=residual_model,
            baseline_spec=baseline_spec,
        )
        layout = getattr(predictor, "layout", None)
        if layout is None:
            raise TypeError("Mixed tensor residual predictor must expose a tensor layout.")
        self.layout = layout
        self.backend = backend
        self.cat_dims = tuple(cat_dims)
        self.material_encoder = material_encoder
        self.structures = tuple(getattr(predictor, "structures", ()))
        if hasattr(predictor, "num_atoms"):
            self.num_atoms = int(predictor.num_atoms)

    def unflatten(self, values: Tensor) -> Tensor:
        """Restore ``[..., num_atoms, 3]`` force or ``[..., 3, 3]`` stress axes."""

        return self.layout.unflatten(values)


def _create_mace(
    quantity: str,
    train_X: Tensor,
    train_Y: Tensor,
    train_Yvar: Tensor | None,
    *,
    structures: Any,
    cat_dims: tuple[int, ...],
    kwargs: dict[str, Any],
) -> ResidualMaterialGPModel:
    energy_module = _load(".mace_residual")
    tensor_module = _load(".mace_tensor_residual")
    mixed_module = import_module("bochan.models.regression.gaussian.deep.mace_mixed")
    multitask_module = import_module(
        "bochan.models.regression.gaussian.deep.mace_multitask"
    )

    model_name = _pop(kwargs, "model_name", energy_module._DEFAULT_MODEL_NAME)
    num_layers = _pop(kwargs, "num_layers", -1)
    pooling = _pop(kwargs, "pooling", "mean")
    head = _pop(kwargs, "head", None)
    latent_dim = _pop(kwargs, "latent_dim", 32)
    fusion = _pop(kwargs, "fusion", "concat")
    projection = _pop(kwargs, "projection", None)
    likelihood = _pop(kwargs, "likelihood", None)
    input_transform = _pop(kwargs, "input_transform", "DEFAULT")
    outcome_transform = _pop(kwargs, "outcome_transform", "DEFAULT")
    target_contract = _pop(kwargs, "target_contract", None)

    material_encoder = energy_module._resolve_encoder(
        _pop(kwargs, "encoder", None),
        model_name=model_name,
        num_layers=num_layers,
        pooling=pooling,
        head=head,
    )
    resolved_structures = energy_module._validate_structure_bank(structures)
    common = dict(
        structures=resolved_structures,
        encoder=material_encoder,
        model_name=model_name,
        num_layers=num_layers,
        pooling=pooling,
        head=head,
        latent_dim=latent_dim,
        fusion=fusion,
        projection=projection,
        likelihood=likelihood,
        input_transform=input_transform,
        outcome_transform=outcome_transform,
    )

    if quantity == "energy":
        if target_contract is not None:
            raise TypeError("MACE energy residual does not currently expose target_contract.")
        _reject_unused(kwargs, backend="mace", quantity=quantity)
        predictor = energy_module.MACEDirectEnergyPredictor(
            material_encoder, resolved_structures
        )
        residual_y = compute_material_residual_targets(train_X, train_Y, predictor)
        residual_model = mixed_module.MACEMixedGPModel(
            train_X,
            residual_y,
            cat_dims,
            train_Yvar,
            **common,
        )
        return MixedScalarResidualMaterialGPModel(
            predictor=predictor,
            residual_model=residual_model,
            backend="mace",
            cat_dims=cat_dims,
            material_encoder=material_encoder,
        )

    num_atoms = _pop(kwargs, "num_atoms", None)
    if quantity == "force":
        predictor = tensor_module.MACEDirectForcePredictor(
            material_encoder,
            resolved_structures,
            num_atoms=num_atoms,
        )
    else:
        if num_atoms is not None:
            raise TypeError("num_atoms is only valid for force residual models.")
        predictor = tensor_module.MACEDirectStressPredictor(
            material_encoder, resolved_structures
        )
    flat_y = predictor.layout.flatten(train_Y, n=train_X.shape[0])
    flat_yvar = (
        None
        if train_Yvar is None
        else predictor.layout.flatten(train_Yvar, n=train_X.shape[0])
    )
    baseline_spec = tensor_module._baseline_spec(
        kind=quantity,
        model_name=model_name,
        target_contract=target_contract,
    )
    residual_y = compute_material_residual_targets(
        train_X,
        flat_y,
        predictor,
        baseline_spec=baseline_spec,
        target_contract=target_contract,
    )
    _reject_unused(kwargs, backend="mace", quantity=quantity)
    residual_model = multitask_module.MACEMixedMultiTaskGPModel(
        train_X,
        residual_y,
        flat_yvar,
        cat_dims=cat_dims,
        **common,
    )
    return MixedTensorResidualMaterialGPModel(
        predictor=predictor,
        residual_model=residual_model,
        backend="mace",
        cat_dims=cat_dims,
        baseline_spec=baseline_spec,
        material_encoder=material_encoder,
    )


def _create_chgnet(
    quantity: str,
    train_X: Tensor,
    train_Y: Tensor,
    train_Yvar: Tensor | None,
    *,
    structures: Any,
    cat_dims: tuple[int, ...],
    kwargs: dict[str, Any],
) -> ResidualMaterialGPModel:
    energy_module = _load(".chgnet_residual")
    tensor_module = _load(".chgnet_tensor_residual")
    gp_module = import_module("bochan.models.regression.gaussian.deep.chgnet")
    multitask_module = import_module(
        "bochan.models.regression.gaussian.deep.chgnet_multitask"
    )

    checkpoint = _pop(kwargs, "checkpoint", None)
    model_name = _pop(kwargs, "model_name", "0.3.0")
    encoder_output_dim = _pop(kwargs, "encoder_output_dim", None)
    latent_dim = _pop(kwargs, "latent_dim", 32)
    fusion = _pop(kwargs, "fusion", "concat")
    projection = _pop(kwargs, "projection", None)
    strict_checkpoint = _pop(kwargs, "strict_checkpoint", True)
    likelihood = _pop(kwargs, "likelihood", None)
    input_transform = _pop(kwargs, "input_transform", "DEFAULT")
    outcome_transform = _pop(kwargs, "outcome_transform", "DEFAULT")
    target_contract = _pop(kwargs, "target_contract", None)

    material_encoder = energy_module._resolve_encoder(
        _pop(kwargs, "encoder", None),
        checkpoint=checkpoint,
        model_name=model_name,
        encoder_output_dim=encoder_output_dim,
        strict_checkpoint=strict_checkpoint,
    )
    resolved_structures = energy_module._validate_structure_bank(structures)
    common = dict(
        structures=resolved_structures,
        encoder=material_encoder,
        model_name=model_name,
        encoder_output_dim=encoder_output_dim,
        latent_dim=latent_dim,
        fusion=fusion,
        projection=projection,
        strict_checkpoint=strict_checkpoint,
        likelihood=likelihood,
        input_transform=input_transform,
        outcome_transform=outcome_transform,
    )

    if quantity == "energy":
        if target_contract is not None:
            raise TypeError(
                "CHGNet energy residual does not currently expose target_contract."
            )
        _reject_unused(kwargs, backend="chgnet", quantity=quantity)
        predictor = energy_module.CHGNetDirectEnergyPredictor(
            material_encoder, resolved_structures
        )
        residual_y = compute_material_residual_targets(train_X, train_Y, predictor)
        residual_model = gp_module.CHGNetMixedGPModel(
            train_X,
            residual_y,
            cat_dims,
            train_Yvar,
            **common,
        )
        return MixedScalarResidualMaterialGPModel(
            predictor=predictor,
            residual_model=residual_model,
            backend="chgnet",
            cat_dims=cat_dims,
            material_encoder=material_encoder,
        )

    num_atoms = _pop(kwargs, "num_atoms", None)
    if quantity == "force":
        predictor = tensor_module.CHGNetDirectForcePredictor(
            material_encoder,
            resolved_structures,
            num_atoms=num_atoms,
        )
    else:
        if num_atoms is not None:
            raise TypeError("num_atoms is only valid for force residual models.")
        predictor = tensor_module.CHGNetDirectStressPredictor(
            material_encoder, resolved_structures
        )
    flat_y = predictor.layout.flatten(train_Y, n=train_X.shape[0])
    flat_yvar = (
        None
        if train_Yvar is None
        else predictor.layout.flatten(train_Yvar, n=train_X.shape[0])
    )
    baseline_spec = tensor_module._baseline_spec(
        kind=quantity,
        model_name=model_name,
        target_contract=target_contract,
    )
    residual_y = compute_material_residual_targets(
        train_X,
        flat_y,
        predictor,
        baseline_spec=baseline_spec,
        target_contract=target_contract,
    )
    _reject_unused(kwargs, backend="chgnet", quantity=quantity)
    residual_model = multitask_module.CHGNetMixedMultiTaskGPModel(
        train_X,
        residual_y,
        flat_yvar,
        cat_dims=cat_dims,
        **common,
    )
    return MixedTensorResidualMaterialGPModel(
        predictor=predictor,
        residual_model=residual_model,
        backend="chgnet",
        cat_dims=cat_dims,
        baseline_spec=baseline_spec,
        material_encoder=material_encoder,
    )


def _create_m3gnet(
    quantity: str,
    train_X: Tensor,
    train_Y: Tensor,
    train_Yvar: Tensor | None,
    *,
    structures: Any,
    cat_dims: tuple[int, ...],
    kwargs: dict[str, Any],
) -> ResidualMaterialGPModel:
    energy_module = _load(".m3gnet_residual")
    tensor_module = _load(".m3gnet_tensor_residual")
    gp_module = import_module("bochan.models.regression.gaussian.deep.m3gnet")
    multitask_module = import_module(
        "bochan.models.regression.gaussian.deep.m3gnet_multitask"
    )

    model_name = _pop(kwargs, "model_name", energy_module._DEFAULT_MODEL_NAME)
    encoder_output_dim = _pop(kwargs, "encoder_output_dim", None)
    latent_dim = _pop(kwargs, "latent_dim", 32)
    fusion = _pop(kwargs, "fusion", "concat")
    projection = _pop(kwargs, "projection", None)
    likelihood = _pop(kwargs, "likelihood", None)
    input_transform = _pop(kwargs, "input_transform", "DEFAULT")
    outcome_transform = _pop(kwargs, "outcome_transform", "DEFAULT")
    target_contract = _pop(kwargs, "target_contract", None)

    material_encoder = energy_module._resolve_encoder(
        _pop(kwargs, "encoder", None),
        model_name=model_name,
        encoder_output_dim=encoder_output_dim,
    )
    resolved_structures = energy_module._validate_structure_bank(structures)
    common = dict(
        structures=resolved_structures,
        encoder=material_encoder,
        model_name=model_name,
        encoder_output_dim=encoder_output_dim,
        latent_dim=latent_dim,
        fusion=fusion,
        projection=projection,
        likelihood=likelihood,
        input_transform=input_transform,
        outcome_transform=outcome_transform,
    )

    if quantity == "energy":
        if target_contract is not None:
            raise TypeError(
                "M3GNet energy residual does not currently expose target_contract."
            )
        _reject_unused(kwargs, backend="m3gnet", quantity=quantity)
        predictor = energy_module.M3GNetDirectPredictor(
            material_encoder, resolved_structures
        )
        residual_y = compute_material_residual_targets(train_X, train_Y, predictor)
        residual_model = gp_module.M3GNetMixedGPModel(
            train_X,
            residual_y,
            cat_dims,
            train_Yvar,
            **common,
        )
        return MixedScalarResidualMaterialGPModel(
            predictor=predictor,
            residual_model=residual_model,
            backend="m3gnet",
            cat_dims=cat_dims,
            material_encoder=material_encoder,
        )

    num_atoms = _pop(kwargs, "num_atoms", None)
    potential = _pop(kwargs, "potential", None)
    calculator = _pop(kwargs, "calculator", None)
    adapter = _pop(kwargs, "adapter", None)
    predictor_kwargs = dict(
        model_name=model_name,
        potential=potential,
        calculator=calculator,
        adapter=adapter,
    )
    if quantity == "force":
        predictor = tensor_module.M3GNetDirectForcePredictor(
            resolved_structures,
            num_atoms=num_atoms,
            **predictor_kwargs,
        )
    else:
        if num_atoms is not None:
            raise TypeError("num_atoms is only valid for force residual models.")
        predictor = tensor_module.M3GNetDirectStressPredictor(
            resolved_structures,
            **predictor_kwargs,
        )
    flat_y = predictor.layout.flatten(train_Y, n=train_X.shape[0])
    flat_yvar = (
        None
        if train_Yvar is None
        else predictor.layout.flatten(train_Yvar, n=train_X.shape[0])
    )
    baseline_spec = tensor_module._baseline_spec(
        kind=quantity,
        model_name=model_name,
        target_contract=target_contract,
    )
    residual_y = compute_material_residual_targets(
        train_X,
        flat_y,
        predictor,
        baseline_spec=baseline_spec,
        target_contract=target_contract,
    )
    _reject_unused(kwargs, backend="m3gnet", quantity=quantity)
    residual_model = multitask_module.M3GNetMixedMultiTaskGPModel(
        train_X,
        residual_y,
        flat_yvar,
        cat_dims=cat_dims,
        **common,
    )
    return MixedTensorResidualMaterialGPModel(
        predictor=predictor,
        residual_model=residual_model,
        backend="m3gnet",
        cat_dims=cat_dims,
        baseline_spec=baseline_spec,
        material_encoder=material_encoder,
    )


def _create_alignn_ff(
    quantity: str,
    train_X: Tensor,
    train_Y: Tensor,
    train_Yvar: Tensor | None,
    *,
    structures: Any,
    cat_dims: tuple[int, ...],
    kwargs: dict[str, Any],
) -> ResidualMaterialGPModel:
    module = _load(".alignn_ff_residual")
    mixed_module = import_module("bochan.models.regression.gaussian.deep.alignn_mixed")
    multitask_module = import_module(
        "bochan.models.regression.gaussian.deep.alignn_multitask"
    )

    structure_graphs = _pop(kwargs, "structure_graphs", None)
    if structure_graphs is None:
        raise ValueError("ALIGNN-FF mixed residual GP requires structure_graphs.")
    if len(structures) != len(structure_graphs):
        raise ValueError(
            "structures and structure_graphs must have the same length and order."
        )

    checkpoint = _pop(kwargs, "checkpoint", None)
    encoder_output_dim = _pop(kwargs, "encoder_output_dim", None)
    encoder_config = _pop(kwargs, "encoder_config", None)
    latent_dim = _pop(kwargs, "latent_dim", 32)
    fusion = _pop(kwargs, "fusion", "concat")
    projection = _pop(kwargs, "projection", None)
    strict_checkpoint = _pop(kwargs, "strict_checkpoint", True)
    likelihood = _pop(kwargs, "likelihood", None)
    input_transform = _pop(kwargs, "input_transform", "DEFAULT")
    outcome_transform = _pop(kwargs, "outcome_transform", "DEFAULT")
    model_name = _pop(kwargs, "model_name", module._DEFAULT_MODEL_NAME)
    calculator = _pop(kwargs, "calculator", None)
    adapter = _pop(kwargs, "adapter", None)
    target_contract = _pop(kwargs, "target_contract", None)
    num_atoms = _pop(kwargs, "num_atoms", None)

    material_encoder = module._resolve_material_encoder(
        _pop(kwargs, "encoder", None),
        checkpoint,
        output_dim=encoder_output_dim,
        config=encoder_config,
        strict_checkpoint=strict_checkpoint,
    )
    predictor_kwargs = dict(
        model_name=model_name,
        calculator=calculator,
        adapter=adapter,
    )
    if quantity == "energy":
        if num_atoms is not None:
            raise TypeError("num_atoms is only valid for force residual models.")
        predictor = module.ALIGNNFFDirectEnergyPredictor(
            structures, **predictor_kwargs
        )
        observed_y = train_Y
        observed_yvar = train_Yvar
    elif quantity == "force":
        predictor = module.ALIGNNFFDirectForcePredictor(
            structures,
            num_atoms=num_atoms,
            **predictor_kwargs,
        )
        observed_y = predictor.layout.flatten(train_Y, n=train_X.shape[0])
        observed_yvar = (
            None
            if train_Yvar is None
            else predictor.layout.flatten(train_Yvar, n=train_X.shape[0])
        )
    else:
        if num_atoms is not None:
            raise TypeError("num_atoms is only valid for force residual models.")
        predictor = module.ALIGNNFFDirectStressPredictor(
            structures, **predictor_kwargs
        )
        observed_y = predictor.layout.flatten(train_Y, n=train_X.shape[0])
        observed_yvar = (
            None
            if train_Yvar is None
            else predictor.layout.flatten(train_Yvar, n=train_X.shape[0])
        )

    baseline_spec = module._baseline_spec(quantity, model_name, target_contract)
    residual_y = compute_material_residual_targets(
        train_X,
        observed_y,
        predictor,
        baseline_spec=baseline_spec,
        target_contract=target_contract,
    )
    _reject_unused(kwargs, backend="alignn-ff", quantity=quantity)
    common = dict(
        structure_graphs=structure_graphs,
        encoder=material_encoder,
        encoder_output_dim=encoder_output_dim,
        encoder_config=encoder_config,
        latent_dim=latent_dim,
        fusion=fusion,
        projection=projection,
        strict_checkpoint=strict_checkpoint,
        likelihood=likelihood,
        input_transform=input_transform,
        outcome_transform=outcome_transform,
    )

    if quantity == "energy":
        residual_model = mixed_module.ALIGNNMixedGPModel(
            train_X,
            residual_y,
            cat_dims,
            observed_yvar,
            **common,
        )
        return MixedScalarResidualMaterialGPModel(
            predictor=predictor,
            residual_model=residual_model,
            backend="alignn-ff",
            cat_dims=cat_dims,
            baseline_spec=baseline_spec,
            material_encoder=material_encoder,
        )

    residual_model = multitask_module.ALIGNNMixedMultiTaskGPModel(
        train_X,
        residual_y,
        observed_yvar,
        cat_dims=cat_dims,
        **common,
    )
    return MixedTensorResidualMaterialGPModel(
        predictor=predictor,
        residual_model=residual_model,
        backend="alignn-ff",
        cat_dims=cat_dims,
        baseline_spec=baseline_spec,
        material_encoder=material_encoder,
    )


def create_mixed_material_residual_gp(
    backend: MaterialMLIPBackend,
    quantity: str,
    train_X: Tensor,
    train_Y: Tensor,
    train_Yvar: Tensor | None = None,
    /,
    *,
    structures: Any,
    cat_dims: Sequence[int],
    **backend_kwargs: Any,
) -> ResidualMaterialGPModel:
    """Create a mixed-input MLIP residual GP for any public MLIP backend.

    ``train_X[:, 0]`` remains the structure selector. ``cat_dims`` identifies
    integer-coded categorical *process* columns. Remaining process columns are
    continuous. Energy uses a scalar mixed GP; force and stress use correlated
    mixed multitask GPs over their flattened physical tensor components.
    """

    resolved_cat_dims = _require_cat_dims(cat_dims)
    kwargs = dict(backend_kwargs)
    if backend == "mace":
        return _create_mace(
            quantity,
            train_X,
            train_Y,
            train_Yvar,
            structures=structures,
            cat_dims=resolved_cat_dims,
            kwargs=kwargs,
        )
    if backend == "chgnet":
        return _create_chgnet(
            quantity,
            train_X,
            train_Y,
            train_Yvar,
            structures=structures,
            cat_dims=resolved_cat_dims,
            kwargs=kwargs,
        )
    if backend == "m3gnet":
        return _create_m3gnet(
            quantity,
            train_X,
            train_Y,
            train_Yvar,
            structures=structures,
            cat_dims=resolved_cat_dims,
            kwargs=kwargs,
        )
    return _create_alignn_ff(
        quantity,
        train_X,
        train_Y,
        train_Yvar,
        structures=structures,
        cat_dims=resolved_cat_dims,
        kwargs=kwargs,
    )


# Backward-compatible scalar class names retained for direct imports.
class MACEMixedResidualGPModel(MixedScalarResidualMaterialGPModel):
    """Compatibility constructor for the MACE energy mixed residual model."""

    def __new__(cls, train_X: Tensor, train_Y: Tensor, cat_dims: Sequence[int], train_Yvar: Tensor | None = None, **kwargs: Any):
        return create_mixed_material_residual_gp(
            "mace", "energy", train_X, train_Y, train_Yvar, cat_dims=cat_dims, **kwargs
        )


class CHGNetMixedResidualGPModel(MixedScalarResidualMaterialGPModel):
    """Compatibility constructor for the CHGNet energy mixed residual model."""

    def __new__(cls, train_X: Tensor, train_Y: Tensor, cat_dims: Sequence[int], train_Yvar: Tensor | None = None, **kwargs: Any):
        return create_mixed_material_residual_gp(
            "chgnet", "energy", train_X, train_Y, train_Yvar, cat_dims=cat_dims, **kwargs
        )


class M3GNetMixedResidualGPModel(MixedScalarResidualMaterialGPModel):
    """Compatibility constructor for the M3GNet energy mixed residual model."""

    def __new__(cls, train_X: Tensor, train_Y: Tensor, cat_dims: Sequence[int], train_Yvar: Tensor | None = None, **kwargs: Any):
        return create_mixed_material_residual_gp(
            "m3gnet", "energy", train_X, train_Y, train_Yvar, cat_dims=cat_dims, **kwargs
        )


__all__ = [
    "CHGNetMixedResidualGPModel",
    "M3GNetMixedResidualGPModel",
    "MACEMixedResidualGPModel",
    "MixedScalarResidualMaterialGPModel",
    "MixedTensorResidualMaterialGPModel",
    "create_mixed_material_residual_gp",
]
