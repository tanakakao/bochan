"""ModelConfig adapters for Gaussian multi-fidelity surrogates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from torch import Tensor

from .correlated import GaussianCorrelatedMultiFidelityGP
from .factory import create_fidelity_surrogate
from .source import GaussianMultiSourceGP, InformationSourceSpec
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
    correlated_outputs: bool = False,
    **model_kwargs: Any,
) -> Any:
    """Create the configured Gaussian multi-fidelity surrogate.

    ``correlated_outputs=True`` keeps the public ``model_type='multifidelity_gp'``
    contract while selecting the Phase 64 Kronecker ICM model. Without the flag,
    the existing single-output / independent multi-output path is unchanged.
    """

    if correlated_outputs:
        return create_configured_correlated_fidelity_surrogate(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            cat_dims=cat_dims,
            fidelity_spec=fidelity_spec,
            fidelity_features=fidelity_features,
            target_fidelities=target_fidelities,
            bounds=bounds,
            input_mode=input_mode,
            **model_kwargs,
        )

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


def create_configured_correlated_fidelity_surrogate(
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
) -> GaussianCorrelatedMultiFidelityGP:
    """Create a correlated Kronecker multi-output multi-fidelity GP."""

    if cat_dims:
        raise NotImplementedError(
            "Phase 64 correlated multi-output MF supports continuous inputs only; "
            "use independent multifidelity_gp for mixed inputs."
        )
    mode = str(input_mode or "normal").lower()
    if mode not in {"normal", "continuous"}:
        raise NotImplementedError(
            "Correlated multi-output MF supports input_mode='normal' only in Phase 64."
        )
    spec = _make_fidelity_spec(
        fidelity_spec=fidelity_spec,
        fidelity_features=fidelity_features,
        target_fidelities=target_fidelities,
    )
    return GaussianCorrelatedMultiFidelityGP(
        train_X=train_X,
        train_Y=train_Y,
        train_Yvar=train_Yvar,
        fidelity_spec=spec,
        bounds=bounds,
        **model_kwargs,
    )


def create_configured_information_source_surrogate(
    train_X: Tensor,
    train_Y: Tensor,
    train_Yvar: Tensor | None = None,
    *,
    cat_dims: Sequence[int] | None = None,
    source_spec: InformationSourceSpec | None = None,
    source_feature: int = -1,
    source_values: Sequence[int] | None = None,
    target_source: int | None = None,
    source_names: Mapping[int, str] | None = None,
    input_mode: str | None = None,
    **model_kwargs: Any,
) -> GaussianMultiSourceGP:
    """Create an unordered discrete multi-information-source ICM GP."""

    if cat_dims:
        raise NotImplementedError(
            "Phase 65 multisource_gp supports continuous design variables plus one "
            "discrete information-source feature; additional categorical inputs are not supported."
        )
    mode = str(input_mode or "normal").lower()
    if mode not in {"normal", "continuous"}:
        raise NotImplementedError(
            "multisource_gp supports input_mode='normal' only in Phase 65."
        )
    if source_spec is not None and any(
        value is not None
        for value in (source_values, target_source, source_names)
    ):
        raise ValueError(
            "Specify source_spec or source_values/target_source/source_names, not both."
        )
    return GaussianMultiSourceGP(
        train_X=train_X,
        train_Y=train_Y,
        train_Yvar=train_Yvar,
        source_spec=source_spec,
        source_feature=source_feature,
        source_values=source_values,
        target_source=target_source,
        source_names=source_names,
        **model_kwargs,
    )


__all__ = [
    "create_configured_correlated_fidelity_surrogate",
    "create_configured_fidelity_surrogate",
    "create_configured_information_source_surrogate",
]
