"""ModelConfig adapter for long-format Gaussian multi-fidelity surrogates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from torch import Tensor

from .factory import create_fidelity_surrogate
from .spec import FidelitySpec, ResolvedFidelitySpec


def _make_fidelity_spec(
    *,
    fidelity_spec: FidelitySpec | ResolvedFidelitySpec | None,
    fidelity_features: Sequence[int] | None,
    target_fidelities: Mapping[int, float] | None,
) -> FidelitySpec | ResolvedFidelitySpec:
    """Normalize ModelConfig shorthand into the shared fidelity contract."""

    has_shorthand = fidelity_features is not None or target_fidelities is not None
    if fidelity_spec is not None and has_shorthand:
        raise ValueError(
            "Specify either fidelity_spec or fidelity_features/target_fidelities, not both."
        )
    if fidelity_spec is not None:
        return fidelity_spec
    if fidelity_features is None:
        raise ValueError(
            "multifidelity_gp requires model_kwargs['fidelity_features'] or "
            "model_kwargs['fidelity_spec']."
        )
    return FidelitySpec(
        fidelity_features=tuple(int(index) for index in fidelity_features),
        target_fidelities=target_fidelities,
    )


def create_configured_fidelity_surrogate(
    train_X: Tensor,
    train_Y: Tensor,
    train_Yvar: Tensor | None = None,
    *,
    cat_dims: Sequence[int] | None = None,
    fidelity_spec: FidelitySpec | ResolvedFidelitySpec | None = None,
    fidelity_features: Sequence[int] | None = None,
    target_fidelities: Mapping[int, float] | None = None,
    bounds: Tensor | None = None,
    input_mode: str | None = None,
    **model_kwargs: Any,
) -> Any:
    """Create a Phase-46 surrogate from public ``ModelConfig.model_kwargs``.

    ``fidelity_features`` and ``target_fidelities`` are convenience fields for
    the high-level API. Internally they are always converted to ``FidelitySpec``.
    The input mode is inferred from ``cat_dims`` unless explicitly supplied.
    """

    spec = _make_fidelity_spec(
        fidelity_spec=fidelity_spec,
        fidelity_features=fidelity_features,
        target_fidelities=target_fidelities,
    )
    mode = input_mode or ("mixed" if cat_dims else "normal")
    return create_fidelity_surrogate(
        train_X,
        train_Y,
        train_Yvar=train_Yvar,
        input_mode=mode,
        cat_dims=cat_dims,
        fidelity_spec=spec,
        bounds=bounds,
        **model_kwargs,
    )


__all__ = ["create_configured_fidelity_surrogate"]
