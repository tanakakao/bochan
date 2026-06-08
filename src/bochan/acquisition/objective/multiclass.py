from __future__ import annotations

import math
from typing import Callable, Literal, Optional, Sequence

import torch
from torch import Tensor

from botorch.acquisition.multi_objective.objective import MCMultiOutputObjective
from botorch.acquisition.objective import MCAcquisitionObjective

from .binary import (
    BinaryClassificationScoreObjective,
    BinaryClassificationScoreObjectiveMixin,
    MultiOutputBinaryClassificationInputPerturbationObjective,
    MultiOutputBinaryClassificationScoreObjective,
    MultiOutputBinaryClassificationScoreObjectiveMixin,
)

RiskType = Optional[Literal["var", "cvar"]]
ClassReductionType = Literal["mean", "sum", "max", "min", "prod"]
MulticlassObjectiveMode = Literal["expected_utility", "target_probability"]


# ============================================================
# Common helpers
# ============================================================


def _validate_n_w_risk(
    *,
    n_w: Optional[int],
    risk_type: RiskType,
    alpha: float,
) -> None:
    if n_w is not None and int(n_w) <= 0:
        raise ValueError("n_w must be a positive integer or None.")

    if risk_type not in (None, "var", "cvar"):
        raise ValueError(f"Unknown risk_type: {risk_type!r}.")

    if risk_type is not None and n_w is None:
        raise ValueError("risk_type is specified, but n_w is None.")

    if risk_type is not None and not (0.0 < float(alpha) <= 1.0):
        raise ValueError("alpha must be in (0, 1].")


def _aggregate_scalar_axis(
    values_w: Tensor,
    *,
    n_w: int,
    risk_type: RiskType,
    alpha: float,
    risk_dim: int,
    maximize: bool = True,
) -> Tensor:
    """Aggregate an explicit InputPerturbation axis."""

    if risk_type is None:
        return values_w.mean(dim=risk_dim)

    descending = not maximize
    sorted_values = torch.sort(values_w, dim=risk_dim, descending=descending).values
    k = max(1, int(math.ceil(int(n_w) * float(alpha))))
    tail = sorted_values.narrow(dim=risk_dim, start=0, length=k)

    if risk_type == "var":
        return tail.select(dim=risk_dim, index=k - 1)

    if risk_type == "cvar":
        return tail.mean(dim=risk_dim)

    raise ValueError(f"Unknown risk_type: {risk_type!r}.")


def _canonicalize_utility_values(
    utility_values: Sequence[float] | Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    utilities = torch.as_tensor(utility_values, device=device, dtype=dtype)
    if utilities.ndim not in (1, 2):
        raise ValueError(
            "utility_values must be 1D [C] or 2D [m, C]. "
            f"Got shape={tuple(utilities.shape)}."
        )
    return utilities


def _as_class_index_list(target_class: int | Sequence[int] | Tensor) -> list[int]:
    if torch.is_tensor(target_class):
        target_class = target_class.detach().cpu().tolist()
    if isinstance(target_class, int):
        return [int(target_class)]
    return [int(i) for i in target_class]


def _select_class_probs(
    probs: Tensor,
    *,
    target_class: int | Sequence[int] | Tensor,
    class_reduction: ClassReductionType = "mean",
) -> Tensor:
    indices = _as_class_index_list(target_class)
    selected = probs[..., indices]

    if selected.shape[-1] == 1:
        return selected.squeeze(-1)

    if class_reduction == "mean":
        return selected.mean(dim=-1)
    if class_reduction == "sum":
        return selected.sum(dim=-1)
    if class_reduction == "max":
        return selected.max(dim=-1).values
    if class_reduction == "min":
        return selected.min(dim=-1).values
    if class_reduction == "prod":
        return selected.prod(dim=-1)

    raise ValueError(f"Unknown class_reduction: {class_reduction!r}.")


def _move_class_dim_to_last(
    values: Tensor,
    *,
    num_classes: Optional[int] = None,
    name: str = "values",
) -> Tensor:
    """Move a class dimension to the last axis when it is not already there.

    Multiclass latent GP posteriors can be represented as class-batched tensors,
    e.g. ``sample_shape x batch_shape x C x q x 1``. Objective code expects
    ``... x C``. This helper keeps already canonical tensors unchanged and moves
    the first matching class dimension otherwise.
    """

    if num_classes is None:
        return values

    c = int(num_classes)
    if values.ndim == 0:
        raise RuntimeError(f"{name}: expected a tensor with a class dimension. Got scalar tensor.")

    out = values
    while out.ndim >= 2 and out.shape[-1] == 1 and out.shape[-2] != c:
        out = out.squeeze(-1)

    if out.shape[-1] == c:
        return out

    for dim in range(out.ndim - 2, -1, -1):
        if out.shape[dim] == c:
            return out.movedim(dim, -1)

    return values


def multiclass_probs_from_logits(
    logits: Tensor,
    *,
    num_classes: Optional[int] = None,
    eps: float = 1e-12,
) -> Tensor:
    """Convert multiclass logits to normalized class probabilities."""

    logits = _move_class_dim_to_last(logits, num_classes=num_classes, name="logits")
    probs = torch.softmax(logits, dim=-1)
    return normalize_multiclass_probs(probs, num_classes=num_classes, eps=eps)


def normalize_multiclass_probs(
    probs: Tensor,
    *,
    num_classes: Optional[int] = None,
    eps: float = 1e-12,
) -> Tensor:
    """Normalize multiclass probability tensors along the class dimension."""

    probs = _move_class_dim_to_last(probs, num_classes=num_classes, name="probs")

    if probs.ndim < 1 or probs.shape[-1] <= 1:
        raise RuntimeError(
            "multiclass probability tensor must have class dim C >= 2. "
            f"Got shape={tuple(probs.shape)}."
        )

    probs = probs.clamp_min(float(eps))
    return probs / probs.sum(dim=-1, keepdim=True).clamp_min(float(eps))


def multiclass_expected_utility_from_probs(
    probs: Tensor,
    utility_values: Sequence[float] | Tensor,
    *,
    num_classes: Optional[int] = None,
    eps: float = 1e-12,
) -> Tensor:
    """Compute expected utility from class probabilities.

    Args:
        probs: Probability tensor with final class dimension, or a class-batched
            tensor that can be canonicalized using ``num_classes``.
        utility_values: Either ``[C]`` shared utilities or ``[m, C]`` utilities
            for multi-output multiclass tensors with shape ``... x m x C``.
        num_classes: Optional class count used to identify a class-batch axis.
        eps: Numerical stability constant.

    Returns:
        Expected utility tensor with class dimension removed.
    """

    probs = normalize_multiclass_probs(probs, num_classes=num_classes, eps=eps)
    utilities = _canonicalize_utility_values(
        utility_values,
        device=probs.device,
        dtype=probs.dtype,
    )

    if utilities.ndim == 1:
        if probs.shape[-1] != utilities.numel():
            raise RuntimeError(
                "Number of classes does not match utility_values length. "
                f"probs.shape[-1]={probs.shape[-1]}, utility_values={utilities.numel()}."
            )
        return (probs * utilities).sum(dim=-1)

    # utilities: [m, C], probs: ... x m x C
    if probs.ndim < 2 or probs.shape[-2:] != utilities.shape:
        raise RuntimeError(
            "2D utility_values requires probs shape (..., m, C). "
            f"Got probs.shape={tuple(probs.shape)}, utility_values.shape={tuple(utilities.shape)}."
        )

    return (probs * utilities).sum(dim=-1)


def multiclass_expected_utility_from_logits(
    logits: Tensor,
    utility_values: Sequence[float] | Tensor,
    *,
    num_classes: Optional[int] = None,
    eps: float = 1e-12,
) -> Tensor:
    """Compute expected utility from multiclass logits."""

    probs = multiclass_probs_from_logits(logits, num_classes=num_classes, eps=eps)
    return multiclass_expected_utility_from_probs(
        probs,
        utility_values=utility_values,
        num_classes=num_classes,
        eps=eps,
    )


def multiclass_target_probability_from_probs(
    probs: Tensor,
    target_class: int | Sequence[int] | Tensor,
    *,
    class_reduction: ClassReductionType = "mean",
    num_classes: Optional[int] = None,
    eps: float = 1e-12,
) -> Tensor:
    """Extract target-class probability from class probabilities."""

    probs = normalize_multiclass_probs(probs, num_classes=num_classes, eps=eps)
    return _select_class_probs(probs, target_class=target_class, class_reduction=class_reduction)


def multiclass_target_probability_from_logits(
    logits: Tensor,
    target_class: int | Sequence[int] | Tensor,
    *,
    class_reduction: ClassReductionType = "mean",
    num_classes: Optional[int] = None,
    eps: float = 1e-12,
) -> Tensor:
    """Extract target-class probability from multiclass logits."""

    probs = multiclass_probs_from_logits(logits, num_classes=num_classes, eps=eps)
    return multiclass_target_probability_from_probs(
        probs,
        target_class=target_class,
        class_reduction=class_reduction,
        num_classes=num_classes,
        eps=eps,
    )


def _looks_like_logits(values: Tensor, *, eps: float) -> bool:
    return bool(values.min() < -float(eps) or values.max() > 1.0 + float(eps))


def _to_probs(
    samples: Tensor,
    *,
    input_is_logits: bool | Literal["auto"] = "auto",
    apply_softmax_if_needed: bool = True,
    num_classes: Optional[int] = None,
    eps: float = 1e-12,
) -> Tensor:
    samples = _move_class_dim_to_last(samples, num_classes=num_classes, name="samples")

    if input_is_logits == "auto":
        use_logits = bool(apply_softmax_if_needed and _looks_like_logits(samples, eps=eps))
    else:
        use_logits = bool(input_is_logits)

    if use_logits:
        return multiclass_probs_from_logits(samples, num_classes=num_classes, eps=eps)

    return normalize_multiclass_probs(samples, num_classes=num_classes, eps=eps)


def _aggregate_input_perturbation_scalar(
    values: Tensor,
    *,
    X: Optional[Tensor],
    n_w: Optional[int],
    risk_type: RiskType,
    alpha: float,
    maximize: bool,
    aggregate_mean_when_no_risk: bool,
    allow_unexpanded: bool,
) -> Tensor:
    if n_w is None or int(n_w) <= 1:
        return values

    if risk_type is None and not aggregate_mean_when_no_risk:
        return values

    n_w_int = int(n_w)

    if X is not None and X.ndim == 2:
        n = int(X.shape[-2])
        if values.shape[-1] == n:
            return values
        if values.ndim >= 2 and values.shape[-2] == n and values.shape[-1] == n_w_int:
            return _aggregate_scalar_axis(
                values,
                n_w=n_w_int,
                risk_type=risk_type,
                alpha=alpha,
                risk_dim=-1,
                maximize=maximize,
            )
        q_like = values.shape[-1]
        if q_like == n * n_w_int:
            values_w = values.reshape(*values.shape[:-1], n, n_w_int)
            return _aggregate_scalar_axis(
                values_w,
                n_w=n_w_int,
                risk_type=risk_type,
                alpha=alpha,
                risk_dim=-1,
                maximize=maximize,
            )
        if allow_unexpanded:
            return values
        raise RuntimeError(
            "Could not aggregate multiclass baseline samples. "
            f"values.shape={tuple(values.shape)}, X.shape={tuple(X.shape)}, n_w={n_w_int}."
        )

    if X is not None and X.ndim >= 3:
        q = int(X.shape[-2])
        q_like = values.shape[-1]
        if q_like == q:
            return values
        if q_like == q * n_w_int:
            values_w = values.reshape(*values.shape[:-1], q, n_w_int)
            return _aggregate_scalar_axis(
                values_w,
                n_w=n_w_int,
                risk_type=risk_type,
                alpha=alpha,
                risk_dim=-1,
                maximize=maximize,
            )
        if allow_unexpanded:
            return values
        raise RuntimeError(
            "Could not aggregate multiclass candidate samples. "
            f"values.shape={tuple(values.shape)}, X.shape={tuple(X.shape)}, n_w={n_w_int}."
        )

    q_expanded = values.shape[-1]
    if q_expanded % n_w_int != 0:
        if allow_unexpanded:
            return values
        raise RuntimeError(
            "values.shape[-1] must be divisible by n_w for InputPerturbation aggregation. "
            f"Got values.shape={tuple(values.shape)}, n_w={n_w_int}."
        )

    q = q_expanded // n_w_int
    values_w = values.reshape(*values.shape[:-1], q, n_w_int)
    return _aggregate_scalar_axis(
        values_w,
        n_w=n_w_int,
        risk_type=risk_type,
        alpha=alpha,
        risk_dim=-1,
        maximize=maximize,
    )


def _aggregate_input_perturbation_multioutput(
    values: Tensor,
    *,
    X: Optional[Tensor],
    n_w: Optional[int],
    risk_type: RiskType,
    alpha: float,
    maximize: bool,
    aggregate_mean_when_no_risk: bool,
    allow_unexpanded: bool,
) -> Tensor:
    if values.ndim < 2:
        raise RuntimeError(
            "multi-output multiclass values must have shape (..., q_like, m). "
            f"Got shape={tuple(values.shape)}."
        )

    if n_w is None or int(n_w) <= 1:
        return values

    if risk_type is None and not aggregate_mean_when_no_risk:
        return values

    n_w_int = int(n_w)
    m = values.shape[-1]

    if X is not None and X.ndim == 2:
        n = int(X.shape[-2])
        if values.shape[-2] == n:
            return values
        if values.ndim >= 3 and values.shape[-3] == n and values.shape[-2] == n_w_int:
            return _aggregate_scalar_axis(
                values,
                n_w=n_w_int,
                risk_type=risk_type,
                alpha=alpha,
                risk_dim=-2,
                maximize=maximize,
            )
        q_like = values.shape[-2]
        if q_like == n * n_w_int:
            values_w = values.reshape(*values.shape[:-2], n, n_w_int, m)
            return _aggregate_scalar_axis(
                values_w,
                n_w=n_w_int,
                risk_type=risk_type,
                alpha=alpha,
                risk_dim=-2,
                maximize=maximize,
            )
        if allow_unexpanded:
            return values
        raise RuntimeError(
            "Could not aggregate multi-output multiclass baseline samples. "
            f"values.shape={tuple(values.shape)}, X.shape={tuple(X.shape)}, n_w={n_w_int}."
        )

    if X is not None and X.ndim >= 3:
        q = int(X.shape[-2])
        q_like = values.shape[-2]
        if q_like == q:
            return values
        if q_like == q * n_w_int:
            values_w = values.reshape(*values.shape[:-2], q, n_w_int, m)
            return _aggregate_scalar_axis(
                values_w,
                n_w=n_w_int,
                risk_type=risk_type,
                alpha=alpha,
                risk_dim=-2,
                maximize=maximize,
            )
        if allow_unexpanded:
            return values
        raise RuntimeError(
            "Could not aggregate multi-output multiclass candidate samples. "
            f"values.shape={tuple(values.shape)}, X.shape={tuple(X.shape)}, n_w={n_w_int}."
        )

    q_expanded = values.shape[-2]
    if q_expanded % n_w_int != 0:
        if allow_unexpanded:
            return values
        raise RuntimeError(
            "values.shape[-2] must be divisible by n_w for InputPerturbation aggregation. "
            f"Got values.shape={tuple(values.shape)}, n_w={n_w_int}."
        )

    q = q_expanded // n_w_int
    values_w = values.reshape(*values.shape[:-2], q, n_w_int, m)
    return _aggregate_scalar_axis(
        values_w,
        n_w=n_w_int,
        risk_type=risk_type,
        alpha=alpha,
        risk_dim=-2,
        maximize=maximize,
    )


# ============================================================
# 1. Single-output multiclass objectives
# ============================================================


class MulticlassExpectedUtilityMCObjective(MCAcquisitionObjective):
    """Convert multiclass posterior samples to expected-utility samples.

    This objective accepts class probabilities or logits. It returns a scalar
    value per candidate point by applying ``utility_values`` to the class
    probabilities.
    """

    def __init__(
        self,
        utility_values: Sequence[float] | Tensor,
        *,
        input_is_logits: bool | Literal["auto"] = "auto",
        apply_softmax_if_needed: bool = True,
        num_classes: Optional[int] = None,
        maximize: bool = True,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        utility_tensor = torch.as_tensor(utility_values, dtype=torch.double)
        if utility_tensor.ndim != 1:
            raise ValueError(
                "MulticlassExpectedUtilityMCObjective requires 1D utility_values [C]. "
                f"Got shape={tuple(utility_tensor.shape)}."
            )
        self.register_buffer("utility_values", utility_tensor)
        self.input_is_logits = input_is_logits
        self.apply_softmax_if_needed = bool(apply_softmax_if_needed)
        self.num_classes = None if num_classes is None else int(num_classes)
        self.maximize = bool(maximize)
        self.eps = float(eps)

    def forward(self, samples: Tensor, X: Optional[Tensor] = None) -> Tensor:
        probs = _to_probs(
            samples,
            input_is_logits=self.input_is_logits,
            apply_softmax_if_needed=self.apply_softmax_if_needed,
            num_classes=self.num_classes,
            eps=self.eps,
        )
        values = multiclass_expected_utility_from_probs(
            probs,
            utility_values=self.utility_values.to(device=probs.device, dtype=probs.dtype),
            num_classes=self.num_classes,
            eps=self.eps,
        )
        return values if self.maximize else -values


class MulticlassTargetProbabilityMCObjective(MCAcquisitionObjective):
    """Convert multiclass posterior samples to target-class probabilities."""

    def __init__(
        self,
        target_class: int | Sequence[int] | Tensor,
        *,
        class_reduction: ClassReductionType = "mean",
        input_is_logits: bool | Literal["auto"] = "auto",
        apply_softmax_if_needed: bool = True,
        num_classes: Optional[int] = None,
        maximize: bool = True,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.target_class = target_class
        self.class_reduction = class_reduction
        self.input_is_logits = input_is_logits
        self.apply_softmax_if_needed = bool(apply_softmax_if_needed)
        self.num_classes = None if num_classes is None else int(num_classes)
        self.maximize = bool(maximize)
        self.eps = float(eps)

    def forward(self, samples: Tensor, X: Optional[Tensor] = None) -> Tensor:
        probs = _to_probs(
            samples,
            input_is_logits=self.input_is_logits,
            apply_softmax_if_needed=self.apply_softmax_if_needed,
            num_classes=self.num_classes,
            eps=self.eps,
        )
        values = multiclass_target_probability_from_probs(
            probs,
            target_class=self.target_class,
            class_reduction=self.class_reduction,
            num_classes=self.num_classes,
            eps=self.eps,
        )
        return values if self.maximize else -values


class MulticlassInputPerturbationExpectedUtilityObjective(MulticlassExpectedUtilityMCObjective):
    """Multiclass expected-utility objective with InputPerturbation aggregation."""

    def __init__(
        self,
        utility_values: Sequence[float] | Tensor,
        *,
        n_w: Optional[int] = None,
        risk_type: RiskType = None,
        alpha: float = 0.5,
        aggregate_mean_when_no_risk: bool = True,
        allow_unexpanded: bool = True,
        input_is_logits: bool | Literal["auto"] = "auto",
        apply_softmax_if_needed: bool = True,
        num_classes: Optional[int] = None,
        maximize: bool = True,
        eps: float = 1e-12,
    ) -> None:
        super().__init__(
            utility_values=utility_values,
            input_is_logits=input_is_logits,
            apply_softmax_if_needed=apply_softmax_if_needed,
            num_classes=num_classes,
            maximize=maximize,
            eps=eps,
        )
        self.n_w = None if n_w is None else int(n_w)
        self.risk_type = risk_type
        self.alpha = float(alpha)
        self.aggregate_mean_when_no_risk = bool(aggregate_mean_when_no_risk)
        self.allow_unexpanded = bool(allow_unexpanded)
        _validate_n_w_risk(n_w=self.n_w, risk_type=self.risk_type, alpha=self.alpha)

    def forward(self, samples: Tensor, X: Optional[Tensor] = None) -> Tensor:
        values = super().forward(samples=samples, X=X)
        return _aggregate_input_perturbation_scalar(
            values,
            X=X,
            n_w=self.n_w,
            risk_type=self.risk_type,
            alpha=self.alpha,
            maximize=True,
            aggregate_mean_when_no_risk=self.aggregate_mean_when_no_risk,
            allow_unexpanded=self.allow_unexpanded,
        )


class MulticlassInputPerturbationTargetProbabilityObjective(MulticlassTargetProbabilityMCObjective):
    """Multiclass target-probability objective with InputPerturbation aggregation."""

    def __init__(
        self,
        target_class: int | Sequence[int] | Tensor,
        *,
        class_reduction: ClassReductionType = "mean",
        n_w: Optional[int] = None,
        risk_type: RiskType = None,
        alpha: float = 0.5,
        aggregate_mean_when_no_risk: bool = True,
        allow_unexpanded: bool = True,
        input_is_logits: bool | Literal["auto"] = "auto",
        apply_softmax_if_needed: bool = True,
        num_classes: Optional[int] = None,
        maximize: bool = True,
        eps: float = 1e-12,
    ) -> None:
        super().__init__(
            target_class=target_class,
            class_reduction=class_reduction,
            input_is_logits=input_is_logits,
            apply_softmax_if_needed=apply_softmax_if_needed,
            num_classes=num_classes,
            maximize=maximize,
            eps=eps,
        )
        self.n_w = None if n_w is None else int(n_w)
        self.risk_type = risk_type
        self.alpha = float(alpha)
        self.aggregate_mean_when_no_risk = bool(aggregate_mean_when_no_risk)
        self.allow_unexpanded = bool(allow_unexpanded)
        _validate_n_w_risk(n_w=self.n_w, risk_type=self.risk_type, alpha=self.alpha)

    def forward(self, samples: Tensor, X: Optional[Tensor] = None) -> Tensor:
        values = super().forward(samples=samples, X=X)
        return _aggregate_input_perturbation_scalar(
            values,
            X=X,
            n_w=self.n_w,
            risk_type=self.risk_type,
            alpha=self.alpha,
            maximize=True,
            aggregate_mean_when_no_risk=self.aggregate_mean_when_no_risk,
            allow_unexpanded=self.allow_unexpanded,
        )


# ============================================================
# 2. Multi-output multiclass objective
# ============================================================


class MultiOutputMulticlassInputPerturbationObjective(MCMultiOutputObjective):
    """Multi-output multiclass objective for qEHVI / qNEHVI style acquisitions.

    Expected sample shapes:
        - ``sample_shape x batch_shape x q_like x C`` for single output
        - ``sample_shape x batch_shape x q_like x m x C`` for multi-output

    Returns:
        - ``sample_shape x batch_shape x q`` or
        - ``sample_shape x batch_shape x q x m``

    The class dimension is reduced by either expected utility or target-class
    probability, and the optional InputPerturbation axis is aggregated after
    that reduction.
    """

    def __init__(
        self,
        *,
        objective_mode: MulticlassObjectiveMode = "expected_utility",
        utility_values: Sequence[float] | Tensor | Sequence[Sequence[float]] | None = None,
        target_class: int | Sequence[int] | Tensor | None = None,
        class_reduction: ClassReductionType = "mean",
        n_w: Optional[int] = None,
        risk_type: RiskType = None,
        alpha: float = 0.5,
        aggregate_mean_when_no_risk: bool = True,
        allow_unexpanded: bool = True,
        input_is_logits: bool | Literal["auto"] = "auto",
        apply_softmax_if_needed: bool = True,
        num_classes: Optional[int] = None,
        maximize: bool = True,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        if objective_mode not in ("expected_utility", "target_probability"):
            raise ValueError("objective_mode must be 'expected_utility' or 'target_probability'.")
        if objective_mode == "expected_utility" and utility_values is None:
            raise ValueError("utility_values is required when objective_mode='expected_utility'.")
        if objective_mode == "target_probability" and target_class is None:
            raise ValueError("target_class is required when objective_mode='target_probability'.")

        self.objective_mode = objective_mode
        self.target_class = target_class
        self.class_reduction = class_reduction
        self.n_w = None if n_w is None else int(n_w)
        self.risk_type = risk_type
        self.alpha = float(alpha)
        self.aggregate_mean_when_no_risk = bool(aggregate_mean_when_no_risk)
        self.allow_unexpanded = bool(allow_unexpanded)
        self.input_is_logits = input_is_logits
        self.apply_softmax_if_needed = bool(apply_softmax_if_needed)
        self.num_classes = None if num_classes is None else int(num_classes)
        self.maximize = bool(maximize)
        self.eps = float(eps)

        if utility_values is None:
            utility_tensor = torch.empty(0, dtype=torch.double)
        else:
            utility_tensor = torch.as_tensor(utility_values, dtype=torch.double)
            if utility_tensor.ndim not in (1, 2):
                raise ValueError(
                    "utility_values must be 1D [C] or 2D [m, C]. "
                    f"Got shape={tuple(utility_tensor.shape)}."
                )
        self.register_buffer("utility_values", utility_tensor)

        _validate_n_w_risk(n_w=self.n_w, risk_type=self.risk_type, alpha=self.alpha)

    def _samples_to_values(self, samples: Tensor) -> Tensor:
        probs = _to_probs(
            samples,
            input_is_logits=self.input_is_logits,
            apply_softmax_if_needed=self.apply_softmax_if_needed,
            num_classes=self.num_classes,
            eps=self.eps,
        )

        if self.objective_mode == "expected_utility":
            values = multiclass_expected_utility_from_probs(
                probs,
                utility_values=self.utility_values.to(device=probs.device, dtype=probs.dtype),
                num_classes=self.num_classes,
                eps=self.eps,
            )
        else:
            values = multiclass_target_probability_from_probs(
                probs,
                target_class=self.target_class,  # type: ignore[arg-type]
                class_reduction=self.class_reduction,
                num_classes=self.num_classes,
                eps=self.eps,
            )

        return values if self.maximize else -values

    def forward(self, samples: Tensor, X: Optional[Tensor] = None) -> Tensor:
        if not torch.is_tensor(samples):
            raise TypeError(f"samples must be a Tensor. Got {type(samples)}.")

        values = self._samples_to_values(samples)

        if values.ndim >= 2 and values.shape[-1] > 1:
            return _aggregate_input_perturbation_multioutput(
                values,
                X=X,
                n_w=self.n_w,
                risk_type=self.risk_type,
                alpha=self.alpha,
                maximize=True,
                aggregate_mean_when_no_risk=self.aggregate_mean_when_no_risk,
                allow_unexpanded=self.allow_unexpanded,
            )

        return _aggregate_input_perturbation_scalar(
            values,
            X=X,
            n_w=self.n_w,
            risk_type=self.risk_type,
            alpha=self.alpha,
            maximize=True,
            aggregate_mean_when_no_risk=self.aggregate_mean_when_no_risk,
            allow_unexpanded=self.allow_unexpanded,
        )


# ============================================================
# 3. Score objective aliases for active-learning score paths
# ============================================================


class MulticlassScoreObjective(BinaryClassificationScoreObjective):
    """Multiclass score objective for already scalarized active-learning scores."""


class MultiOutputMulticlassScoreObjective(MultiOutputBinaryClassificationScoreObjective):
    """Multi-output multiclass score objective for already scalarized scores."""


class MultiOutputMulticlassScoreObjectiveMixin(MultiOutputBinaryClassificationScoreObjectiveMixin):
    """Mixin for applying multi-output multiclass score objectives."""


class MulticlassScoreObjectiveMixin(BinaryClassificationScoreObjectiveMixin):
    """Mixin for applying multiclass score objectives."""


# Backward-compatible naming for score-only input perturbation objective.
MultiOutputMulticlassScoreInputPerturbationObjective = MultiOutputBinaryClassificationInputPerturbationObjective


__all__ = [
    "RiskType",
    "ClassReductionType",
    "MulticlassObjectiveMode",
    "normalize_multiclass_probs",
    "multiclass_probs_from_logits",
    "multiclass_expected_utility_from_probs",
    "multiclass_expected_utility_from_logits",
    "multiclass_target_probability_from_probs",
    "multiclass_target_probability_from_logits",
    "MulticlassExpectedUtilityMCObjective",
    "MulticlassTargetProbabilityMCObjective",
    "MulticlassInputPerturbationExpectedUtilityObjective",
    "MulticlassInputPerturbationTargetProbabilityObjective",
    "MultiOutputMulticlassInputPerturbationObjective",
    "MulticlassScoreObjective",
    "MultiOutputMulticlassScoreObjective",
    "MulticlassScoreObjectiveMixin",
    "MultiOutputMulticlassScoreObjectiveMixin",
    "MultiOutputMulticlassScoreInputPerturbationObjective",
]
