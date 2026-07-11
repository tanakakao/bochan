"""InputPerturbation support for multiclass hypervolume acquisitions.

BoTorch one-to-many input transforms expand a raw q-batch from ``q`` to
``q * n_w``. Hypervolume acquisitions enumerate all non-empty subsets of the
objective q-batch, so passing the expanded q through unchanged has exponential
memory cost. For example, ``q=3`` and ``n_w=8`` becomes 24 effective points and
requires up to ``2**24 - 1`` subset terms.

This module detects the perturbation count from the model input transform,
aggregates the default multiclass objective back to the raw q-batch, and checks
the objective q dimension before BoTorch starts subset enumeration.
"""

from __future__ import annotations

import math
from functools import wraps
from typing import Any, Literal, Optional, TypeVar

import torch
from botorch.acquisition.multi_objective.objective import MCMultiOutputObjective
from torch import Tensor

RiskType = Optional[Literal["var", "cvar"]]
AcquisitionType = TypeVar("AcquisitionType", bound=type)


def infer_input_perturbation_n_w(model: Any) -> int | None:
    """Infer ``n_w`` from InputPerturbation-like modules on ``model``.

    The implementation intentionally relies on the public ``perturbation_set``
    tensor instead of a concrete BoTorch class check, so chained and supported
    custom transforms are supported as well.
    """
    transform = getattr(model, "input_transform", None)
    if transform is None:
        return None

    modules = list(transform.modules()) if hasattr(transform, "modules") else [transform]
    counts: set[int] = set()
    for module in modules:
        perturbation_set = getattr(module, "perturbation_set", None)
        if torch.is_tensor(perturbation_set) and perturbation_set.ndim >= 2:
            count = int(perturbation_set.shape[-2])
            if count > 0:
                counts.add(count)

    if len(counts) == 0:
        return None
    if len(counts) > 1:
        raise ValueError(
            "Multiple InputPerturbation transforms with different n_w values "
            f"were found: {sorted(counts)}. Specify input_perturbation_n_w explicitly."
        )
    return next(iter(counts))


def _aggregate_lower_tail(
    values_w: Tensor,
    *,
    n_w: int,
    risk_type: RiskType,
    alpha: float,
) -> Tensor:
    """Aggregate the perturbation axis for a maximization objective."""
    if risk_type is None:
        return values_w.mean(dim=-2)

    if risk_type not in ("var", "cvar"):
        raise ValueError(f"Unknown input perturbation risk type: {risk_type!r}.")
    if not (0.0 < float(alpha) <= 1.0):
        raise ValueError(f"input perturbation risk alpha must be in (0, 1], got {alpha}.")

    k = max(1, int(math.ceil(int(n_w) * float(alpha))))
    lower_tail = torch.sort(values_w, dim=-2, descending=False).values.narrow(
        dim=-2,
        start=0,
        length=k,
    )
    if risk_type == "var":
        return lower_tail.select(dim=-2, index=k - 1)
    return lower_tail.mean(dim=-2)


class InputPerturbationMultiOutputObjectiveAdapter(MCMultiOutputObjective):
    """Aggregate a multi-output objective from ``q*n_w`` back to raw ``q``.

    The wrapped objective must return ``sample_shape x batch_shape x q_like x m``.
    It remains responsible for converting multiclass probabilities or logits to
    the requested target-class probability / utility values.
    """

    def __init__(
        self,
        objective: MCMultiOutputObjective,
        *,
        n_w: int,
        risk_type: RiskType = None,
        alpha: float = 0.5,
    ) -> None:
        super().__init__()
        if int(n_w) <= 0:
            raise ValueError(f"n_w must be positive, got {n_w}.")
        self.objective = objective
        self.n_w = int(n_w)
        self.risk_type = risk_type
        self.alpha = float(alpha)
        self._verify_output_shape = False
        self._bochan_aggregates_input_perturbation = True

    def forward(self, samples: Tensor, X: Tensor | None = None) -> Tensor:
        values = self.objective(samples, X=X)
        if values.ndim < 2:
            raise RuntimeError(
                "A multi-output hypervolume objective must return shape "
                f"(..., q_like, m), got {tuple(values.shape)}."
            )

        q_like = int(values.shape[-2])
        if X is not None:
            raw_q = int(X.shape[-2])
            if q_like == raw_q:
                return values
            expected_q = raw_q * self.n_w
            if q_like != expected_q:
                raise RuntimeError(
                    "InputPerturbation objective could not align its q dimension. "
                    f"raw q={raw_q}, n_w={self.n_w}, expected q_like={expected_q}, "
                    f"got objective shape={tuple(values.shape)}."
                )
        else:
            if q_like % self.n_w != 0:
                raise RuntimeError(
                    "InputPerturbation objective q dimension must be divisible by n_w "
                    f"when X is unavailable. Got q_like={q_like}, n_w={self.n_w}, "
                    f"shape={tuple(values.shape)}."
                )
            raw_q = q_like // self.n_w

        m = int(values.shape[-1])
        values_w = values.reshape(*values.shape[:-2], raw_q, self.n_w, m)
        return _aggregate_lower_tail(
            values_w,
            n_w=self.n_w,
            risk_type=self.risk_type,
            alpha=self.alpha,
        )


def validate_hypervolume_objective_q(
    objective_values: Tensor,
    X: Tensor | None,
) -> None:
    """Stop before exponential qEHVI subset allocation when q is not reduced."""
    if X is None or objective_values.ndim < 2:
        return
    raw_q = int(X.shape[-2])
    objective_q = int(objective_values.shape[-2])
    if objective_q != raw_q:
        raise RuntimeError(
            "The multiclass hypervolume objective did not reduce the one-to-many "
            "InputPerturbation dimension before qEHVI subset enumeration. "
            f"Raw X q={raw_q}, objective q={objective_q}, "
            f"objective shape={tuple(objective_values.shape)}. Pass an "
            "InputPerturbation-aware objective or leave objective=None so bochan "
            "can infer n_w from model.input_transform."
        )


def configure_multiclass_hypervolume_input_perturbation(
    acquisition_cls: AcquisitionType,
    *,
    default_objective_type: type,
) -> AcquisitionType:
    """Patch a multiclass qEHVI class with automatic perturbation aggregation.

    Only bochan's built-in default multiclass objective is wrapped automatically.
    Explicit custom objectives are left unchanged, but the q-dimension guard still
    prevents an accidental exponential allocation.
    """
    if getattr(acquisition_cls, "_bochan_input_perturbation_patched", False):
        return acquisition_cls

    original_init = acquisition_cls.__init__
    original_compute_qehvi = acquisition_cls._compute_qehvi

    @wraps(original_init)
    def supported_init(self, model, *args, **kwargs):
        explicit_n_w = kwargs.pop("input_perturbation_n_w", None)
        risk_type = kwargs.pop("input_perturbation_risk_type", None)
        risk_alpha = kwargs.pop("input_perturbation_risk_alpha", 0.5)

        original_init(self, model, *args, **kwargs)

        inferred_n_w = infer_input_perturbation_n_w(model)
        n_w = inferred_n_w if explicit_n_w is None else int(explicit_n_w)
        self.input_perturbation_n_w = n_w

        if (
            n_w is not None
            and int(n_w) > 1
            and isinstance(self.objective, default_objective_type)
            and not getattr(
                self.objective,
                "_bochan_aggregates_input_perturbation",
                False,
            )
        ):
            self.objective = InputPerturbationMultiOutputObjectiveAdapter(
                self.objective,
                n_w=int(n_w),
                risk_type=risk_type,
                alpha=float(risk_alpha),
            )

    def supported_compute_qehvi(self, samples: Tensor, X: Tensor | None = None):
        objective_values = self.objective(samples, X=X)
        validate_hypervolume_objective_q(objective_values, X)
        return original_compute_qehvi(self, samples=samples, X=X)

    acquisition_cls.__init__ = supported_init
    acquisition_cls._compute_qehvi = supported_compute_qehvi
    acquisition_cls._bochan_input_perturbation_patched = True
    acquisition_cls._bochan_original_init = original_init
    acquisition_cls._bochan_original_compute_qehvi = original_compute_qehvi
    return acquisition_cls


__all__ = [
    "InputPerturbationMultiOutputObjectiveAdapter",
    "infer_input_perturbation_n_w",
    "configure_multiclass_hypervolume_input_perturbation",
    "validate_hypervolume_objective_q",
]
