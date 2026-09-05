"""Unified construction entry point for Gaussian multi-fidelity surrogates."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

from torch import Tensor

from .spec import FidelitySpec, ResolvedFidelitySpec

if TYPE_CHECKING:
    from bochan.models.regression.gaussian.long_multifidelity import (
        GaussianMixedMultiFidelityGP,
        GaussianMultiFidelityGP,
    )

    FidelitySurrogate = GaussianMultiFidelityGP | GaussianMixedMultiFidelityGP
else:
    FidelitySurrogate = Any

FidelityInputMode = Literal["continuous", "mixed"]


def _normalize_input_mode(input_mode: str) -> FidelityInputMode:
    mode = str(input_mode).strip().lower()
    aliases = {
        "continuous": "continuous",
        "normal": "continuous",
        "mixed": "mixed",
    }
    try:
        return aliases[mode]  # type: ignore[return-value]
    except KeyError as exc:
        raise ValueError(
            "input_mode must be one of {'continuous', 'normal', 'mixed'}."
        ) from exc


def create_fidelity_surrogate(
    train_X: Tensor,
    train_Y: Tensor,
    train_Yvar: Tensor | None = None,
    *,
    input_mode: str = "continuous",
    cat_dims: Sequence[int] | None = None,
    fidelity_spec: FidelitySpec | ResolvedFidelitySpec,
    bounds: Tensor | None = None,
    **model_kwargs: Any,
) -> FidelitySurrogate:
    """Create a long-format Gaussian multi-fidelity surrogate.

    Parameters
    ----------
    train_X:
        Public training inputs. Fidelity must already be represented by one
        feature column.
    train_Y:
        Scalar observations with shape ``[n, 1]``.
    train_Yvar:
        Optional known observation variance with the same shape as ``train_Y``.
    input_mode:
        ``"continuous"`` / ``"normal"`` for continuous design variables or
        ``"mixed"`` for continuous + categorical design variables.
    cat_dims:
        Categorical feature indices for mixed inputs. Forbidden for continuous
        inputs and required for mixed inputs.
    fidelity_spec:
        Shared fidelity-axis contract.
    bounds:
        Optional public input bounds used for fidelity-target validation.
    model_kwargs:
        Additional keyword arguments forwarded to the selected concrete model.
    """

    mode = _normalize_input_mode(input_mode)

    if mode == "continuous":
        if cat_dims is not None and len(tuple(cat_dims)) > 0:
            raise ValueError("cat_dims is only valid when input_mode='mixed'.")
        from bochan.models.regression.gaussian.long_multifidelity import (
            GaussianMultiFidelityGP,
        )

        return GaussianMultiFidelityGP(
            train_X,
            train_Y,
            train_Yvar=train_Yvar,
            fidelity_spec=fidelity_spec,
            bounds=bounds,
            **model_kwargs,
        )

    if cat_dims is None or len(tuple(cat_dims)) == 0:
        raise ValueError("cat_dims is required when input_mode='mixed'.")

    from bochan.models.regression.gaussian.long_multifidelity import (
        GaussianMixedMultiFidelityGP,
    )

    return GaussianMixedMultiFidelityGP(
        train_X,
        train_Y,
        train_Yvar=train_Yvar,
        cat_dims=tuple(cat_dims),
        fidelity_spec=fidelity_spec,
        bounds=bounds,
        **model_kwargs,
    )


__all__ = ["FidelityInputMode", "create_fidelity_surrogate"]
