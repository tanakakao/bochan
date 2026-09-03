"""Backend-neutral factories for MLIP property predictors and residual GPs."""

from __future__ import annotations

from importlib import import_module
from typing import Any, Literal, cast

from torch import Tensor

from .factory import MaterialMLIPBackend, normalize_material_backend

MaterialQuantity = Literal["energy", "force", "stress"]
SUPPORTED_MATERIAL_QUANTITIES: tuple[MaterialQuantity, ...] = (
    "energy",
    "force",
    "stress",
)

_RESIDUAL_IMPORTS: dict[tuple[MaterialMLIPBackend, MaterialQuantity], tuple[str, str]] = {
    ("mace", "energy"): (".mace_residual", "MACEResidualGPModel"),
    ("mace", "force"): (".mace_tensor_residual", "MACEForceResidualGPModel"),
    ("mace", "stress"): (".mace_tensor_residual", "MACEStressResidualGPModel"),
    ("chgnet", "energy"): (".chgnet_residual", "CHGNetResidualGPModel"),
    ("chgnet", "force"): (".chgnet_tensor_residual", "CHGNetForceResidualGPModel"),
    ("chgnet", "stress"): (".chgnet_tensor_residual", "CHGNetStressResidualGPModel"),
    ("m3gnet", "energy"): (".m3gnet_residual", "M3GNetResidualGPModel"),
    ("m3gnet", "force"): (".m3gnet_tensor_residual", "M3GNetForceResidualGPModel"),
    ("m3gnet", "stress"): (".m3gnet_tensor_residual", "M3GNetStressResidualGPModel"),
    ("alignn-ff", "energy"): (".alignn_ff_residual", "ALIGNNFFEnergyResidualGPModel"),
    ("alignn-ff", "force"): (".alignn_ff_residual", "ALIGNNFFForceResidualGPModel"),
    ("alignn-ff", "stress"): (".alignn_ff_residual", "ALIGNNFFStressResidualGPModel"),
}


def normalize_material_quantity(quantity: str) -> MaterialQuantity:
    """Normalize a supported physical target name."""

    if not isinstance(quantity, str) or not quantity.strip():
        raise ValueError("quantity must be a non-empty string.")
    normalized = quantity.strip().lower()
    if normalized not in SUPPORTED_MATERIAL_QUANTITIES:
        supported = ", ".join(SUPPORTED_MATERIAL_QUANTITIES)
        raise ValueError(
            f"Unsupported material quantity {quantity!r}. Supported quantities: {supported}."
        )
    return cast(MaterialQuantity, normalized)


def _load(module_name: str) -> Any:
    return import_module(module_name, package=__package__)


def _pop(kwargs: dict[str, Any], name: str, default: Any) -> Any:
    return kwargs.pop(name, default)


def _reject_unused(kwargs: dict[str, Any], *, backend: str, quantity: str) -> None:
    if kwargs:
        names = ", ".join(sorted(kwargs))
        raise TypeError(
            f"Unsupported direct-predictor arguments for {backend}/{quantity}: {names}."
        )


def _create_mace_predictor(
    quantity: MaterialQuantity,
    structures: Any,
    kwargs: dict[str, Any],
) -> Any:
    energy_module = _load(".mace_residual")
    model_name = _pop(kwargs, "model_name", energy_module._DEFAULT_MODEL_NAME)
    encoder = energy_module._resolve_encoder(
        _pop(kwargs, "encoder", None),
        model_name=model_name,
        num_layers=_pop(kwargs, "num_layers", -1),
        pooling=_pop(kwargs, "pooling", "mean"),
        head=_pop(kwargs, "head", None),
    )
    if quantity == "energy":
        _reject_unused(kwargs, backend="mace", quantity=quantity)
        return energy_module.MACEDirectEnergyPredictor(encoder, structures)
    tensor_module = _load(".mace_tensor_residual")
    if quantity == "force":
        num_atoms = _pop(kwargs, "num_atoms", None)
        _reject_unused(kwargs, backend="mace", quantity=quantity)
        return tensor_module.MACEDirectForcePredictor(
            encoder, structures, num_atoms=num_atoms
        )
    _reject_unused(kwargs, backend="mace", quantity=quantity)
    return tensor_module.MACEDirectStressPredictor(encoder, structures)


def _create_chgnet_predictor(
    quantity: MaterialQuantity,
    structures: Any,
    kwargs: dict[str, Any],
) -> Any:
    energy_module = _load(".chgnet_residual")
    encoder = energy_module._resolve_encoder(
        _pop(kwargs, "encoder", None),
        checkpoint=_pop(kwargs, "checkpoint", None),
        model_name=_pop(kwargs, "model_name", "0.3.0"),
        encoder_output_dim=_pop(kwargs, "encoder_output_dim", None),
        strict_checkpoint=_pop(kwargs, "strict_checkpoint", True),
    )
    if quantity == "energy":
        _reject_unused(kwargs, backend="chgnet", quantity=quantity)
        return energy_module.CHGNetDirectEnergyPredictor(encoder, structures)
    tensor_module = _load(".chgnet_tensor_residual")
    if quantity == "force":
        num_atoms = _pop(kwargs, "num_atoms", None)
        _reject_unused(kwargs, backend="chgnet", quantity=quantity)
        return tensor_module.CHGNetDirectForcePredictor(
            encoder, structures, num_atoms=num_atoms
        )
    _reject_unused(kwargs, backend="chgnet", quantity=quantity)
    return tensor_module.CHGNetDirectStressPredictor(encoder, structures)


def _create_m3gnet_predictor(
    quantity: MaterialQuantity,
    structures: Any,
    kwargs: dict[str, Any],
) -> Any:
    if quantity == "energy":
        module = _load(".m3gnet_residual")
        encoder = module._resolve_encoder(
            _pop(kwargs, "encoder", None),
            model_name=_pop(kwargs, "model_name", module._DEFAULT_MODEL_NAME),
            encoder_output_dim=_pop(kwargs, "encoder_output_dim", None),
        )
        _reject_unused(kwargs, backend="m3gnet", quantity=quantity)
        return module.M3GNetDirectPredictor(encoder, structures)
    module = _load(".m3gnet_tensor_residual")
    class_name = (
        "M3GNetDirectForcePredictor"
        if quantity == "force"
        else "M3GNetDirectStressPredictor"
    )
    predictor_class = getattr(module, class_name)
    return predictor_class(structures, **kwargs)


def _create_alignn_ff_predictor(
    quantity: MaterialQuantity,
    structures: Any,
    kwargs: dict[str, Any],
) -> Any:
    module = _load(".alignn_ff_residual")
    class_name = {
        "energy": "ALIGNNFFDirectEnergyPredictor",
        "force": "ALIGNNFFDirectForcePredictor",
        "stress": "ALIGNNFFDirectStressPredictor",
    }[quantity]
    predictor_class = getattr(module, class_name)
    return predictor_class(structures, **kwargs)


def create_direct_material_predictor(
    backend: str,
    quantity: str,
    /,
    *,
    structures: Any,
    **backend_kwargs: Any,
) -> Any:
    """Create a pretrained direct predictor for Energy, Force, or Stress.

    The factory preserves backend-specific configuration rather than flattening
    every implementation into one lowest-common-denominator constructor. MACE,
    CHGNet, and M3GNet scalar predictors resolve their existing frozen encoders;
    ASE-backed M3GNet tensor predictors and ALIGNN-FF predictors receive their
    calculator/potential options unchanged.
    """

    resolved_backend = normalize_material_backend(backend)
    resolved_quantity = normalize_material_quantity(quantity)
    kwargs = dict(backend_kwargs)
    if resolved_backend == "mace":
        return _create_mace_predictor(resolved_quantity, structures, kwargs)
    if resolved_backend == "chgnet":
        return _create_chgnet_predictor(resolved_quantity, structures, kwargs)
    if resolved_backend == "m3gnet":
        return _create_m3gnet_predictor(resolved_quantity, structures, kwargs)
    return _create_alignn_ff_predictor(resolved_quantity, structures, kwargs)


def create_material_residual_gp(
    backend: str,
    quantity: str,
    train_X: Tensor,
    train_Y: Tensor,
    train_Yvar: Tensor | None = None,
    /,
    *,
    structures: Any,
    **backend_kwargs: Any,
) -> Any:
    """Create the backend/quantity-specific residual GP model.

    All backend-specific model arguments are forwarded unchanged. ALIGNN-FF is
    intentionally stricter: residual GPs require ``structure_graphs`` in
    addition to raw ``structures`` because the physical baseline uses the ASE
    calculator while the GP correction uses the existing ALIGNN graph encoder.
    """

    resolved_backend = normalize_material_backend(backend)
    resolved_quantity = normalize_material_quantity(quantity)
    if resolved_backend == "alignn-ff" and "structure_graphs" not in backend_kwargs:
        raise ValueError(
            "ALIGNN-FF residual GP requires structure_graphs matching structures in length and order."
        )
    module_name, class_name = _RESIDUAL_IMPORTS[(resolved_backend, resolved_quantity)]
    module = _load(module_name)
    model_class = getattr(module, class_name, None)
    if not isinstance(model_class, type):
        raise RuntimeError(
            f"Material backend {resolved_backend!r} does not expose expected residual class {class_name}."
        )
    return model_class(
        train_X,
        train_Y,
        train_Yvar,
        structures=structures,
        **backend_kwargs,
    )


__all__ = [
    "MaterialQuantity",
    "SUPPORTED_MATERIAL_QUANTITIES",
    "create_direct_material_predictor",
    "create_material_residual_gp",
    "normalize_material_quantity",
]
