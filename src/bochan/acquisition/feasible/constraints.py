from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional, Sequence, Union

import torch
from torch import Tensor


ConstraintSense = Literal["ge", "le", "eq"]
OrdinalRankSense = Literal["ge", "le", "eq"]
PerturbationReduction = Literal["mean", "min", "max", "prod"]
OutputKey = Union[int, str]


@dataclass(frozen=True)
class FeasibilityConstraintSpec:
    """Feasible acquisition 用の outcome constraint 定義。

    ``target_class`` / ``target_classes`` を指定しない場合は、posterior sample
    の1出力値をしきい値判定する。指定した場合は、対象 output の class
    probability から ``P(class == target_class)`` または
    ``P(class in target_classes)`` を計算してしきい値判定する。

    BoTorch 標準 ``constraints=`` は sample だけを受け取るため、class
    probability 制約は model access がある ``outcome_constraint_config`` /
    ``FeasibilityWeightedAcquisition`` 経由で使うのが基本。
    """

    output: OutputKey
    threshold: float
    sense: ConstraintSense = "ge"
    margin: float = 0.0
    scale: float = 1.0
    target_class: int | None = None
    target_classes: Sequence[int] | None = None

    def __post_init__(self) -> None:
        if self.sense not in {"ge", "le", "eq"}:
            raise ValueError("sense must be one of 'ge', 'le', or 'eq'.")
        if float(self.scale) <= 0.0:
            raise ValueError("scale must be positive.")
        if self.sense == "eq" and float(self.margin) < 0.0:
            raise ValueError("margin must be non-negative for equality constraints.")
        if self.target_class is not None and self.target_classes is not None:
            raise ValueError("Specify either target_class or target_classes, not both.")
        if self.target_class is not None:
            if int(self.target_class) < 0:
                raise ValueError("target_class must be non-negative.")
            object.__setattr__(self, "target_class", int(self.target_class))
        if self.target_classes is not None:
            classes = [int(cls) for cls in self.target_classes]
            if len(classes) == 0:
                raise ValueError("target_classes must not be empty.")
            if any(cls < 0 for cls in classes):
                raise ValueError("target_classes must contain non-negative class indices.")
            object.__setattr__(self, "target_classes", tuple(classes))

    @property
    def has_target_classes(self) -> bool:
        return self.target_class is not None or self.target_classes is not None

    @property
    def resolved_target_classes(self) -> tuple[int, ...]:
        if self.target_class is not None:
            return (int(self.target_class),)
        if self.target_classes is not None:
            return tuple(int(cls) for cls in self.target_classes)
        return ()


@dataclass(frozen=True)
class OrdinalRankConstraintSpec:
    """Ordinal rank probability を使う feasible constraint 定義。

    例:
        ``OrdinalRankConstraintSpec("quality_rank", rank=2, sense="ge", probability_threshold=0.8)``
        は ``P(y >= 2) >= 0.8`` を feasible とする。
    """

    output: OutputKey
    rank: int
    sense: OrdinalRankSense = "ge"
    probability_threshold: float = 0.8
    scale: float = 1.0

    def __post_init__(self) -> None:
        if self.sense not in {"ge", "le", "eq"}:
            raise ValueError("sense must be one of 'ge', 'le', or 'eq'.")
        if int(self.rank) < 0:
            raise ValueError("rank must be non-negative.")
        p = float(self.probability_threshold)
        if not (0.0 <= p <= 1.0):
            raise ValueError("probability_threshold must be in [0, 1].")
        if float(self.scale) <= 0.0:
            raise ValueError("scale must be positive.")
        object.__setattr__(self, "rank", int(self.rank))


def normalize_output_index(
    output: OutputKey,
    *,
    output_names: Optional[Sequence[str]] = None,
) -> int:
    """出力 index / name を整数 index に正規化する。"""

    if isinstance(output, str):
        if output_names is None:
            raise ValueError("output_names is required when output is a string.")
        names = list(output_names)
        if output not in names:
            raise KeyError(f"Unknown output name {output!r}. Available={names}.")
        return names.index(output)

    idx = int(output)
    if idx < 0:
        raise IndexError(f"output index must be non-negative. Got {idx}.")
    return idx


def constraint_value_from_output(y: Tensor, spec: FeasibilityConstraintSpec) -> Tensor:
    """単一出力値 `y` から BoTorch 形式の constraint value を計算する。

    戻り値は feasible なときに ``<= 0`` になる。
    """

    threshold = float(spec.threshold)
    scale = float(spec.scale)

    if spec.sense == "ge":
        value = threshold - y
    elif spec.sense == "le":
        value = y - threshold
    elif spec.sense == "eq":
        value = torch.abs(y - threshold) - float(spec.margin)
    else:
        raise ValueError(f"Unknown sense={spec.sense!r}.")

    return value / scale


def class_probability_from_probs(probs: Tensor, spec: FeasibilityConstraintSpec) -> Tensor:
    """class probability tensor から指定クラスの確率を取り出す。"""

    classes = spec.resolved_target_classes
    if len(classes) == 0:
        raise ValueError("target_class or target_classes is required.")
    if probs.ndim < 1:
        raise ValueError("probs must have a final class dimension.")
    num_classes = int(probs.shape[-1])
    if any(cls >= num_classes for cls in classes):
        raise IndexError(
            f"target classes {classes} are out of range for probs.shape={tuple(probs.shape)}."
        )
    class_idx = torch.as_tensor(classes, dtype=torch.long, device=probs.device)
    selected = probs.index_select(dim=-1, index=class_idx)
    return selected.sum(dim=-1)


def constraint_value_from_class_probs(
    probs: Tensor,
    spec: FeasibilityConstraintSpec,
) -> Tensor:
    """class probabilities から BoTorch 形式の constraint value を計算する。"""

    p = class_probability_from_probs(probs, spec)
    return constraint_value_from_output(p, spec)


def ordinal_rank_probability(probs: Tensor, spec: OrdinalRankConstraintSpec) -> Tensor:
    """ordinal class probabilities から rank 条件の確率を計算する。"""

    if probs.ndim < 1:
        raise ValueError("probs must have at least one class dimension.")
    if spec.rank >= probs.shape[-1]:
        raise IndexError(
            f"rank={spec.rank} is out of range for probs.shape={tuple(probs.shape)}."
        )

    if spec.sense == "ge":
        return probs[..., spec.rank :].sum(dim=-1)
    if spec.sense == "le":
        return probs[..., : spec.rank + 1].sum(dim=-1)
    if spec.sense == "eq":
        return probs[..., spec.rank]

    raise ValueError(f"Unknown sense={spec.sense!r}.")


def constraint_value_from_ordinal_probs(
    probs: Tensor,
    spec: OrdinalRankConstraintSpec,
) -> Tensor:
    """ordinal probabilities から BoTorch 形式の constraint value を計算する。"""

    p_rank = ordinal_rank_probability(probs, spec)
    value = float(spec.probability_threshold) - p_rank
    return value / float(spec.scale)


def reduce_input_perturbation_values(
    value: Tensor,
    *,
    n_w: int | None,
    reduction: PerturbationReduction = "mean",
) -> Tensor:
    """Reduce the perturbation-expanded q dimension back to the original q."""

    if n_w is None or int(n_w) <= 1:
        return value
    n_w = int(n_w)
    if reduction not in {"mean", "min", "max", "prod"}:
        raise ValueError("reduction must be 'mean', 'min', 'max', or 'prod'.")
    if value.ndim < 1:
        return value
    q_times_n_w = int(value.shape[-1])
    if q_times_n_w % n_w != 0:
        raise RuntimeError(
            "Cannot reduce input-perturbation constraint values because the "
            f"last dimension {q_times_n_w} is not divisible by n_w={n_w}. "
            f"value.shape={tuple(value.shape)}."
        )
    q = q_times_n_w // n_w
    reshaped = value.reshape(*value.shape[:-1], q, n_w)
    if reduction == "mean":
        return reshaped.mean(dim=-1)
    if reduction == "min":
        return reshaped.min(dim=-1).values
    if reduction == "max":
        return reshaped.max(dim=-1).values
    if reduction == "prod":
        return reshaped.prod(dim=-1)
    raise ValueError(f"Unknown reduction={reduction!r}.")


def wrap_sample_constraint_for_input_perturbation(
    constraint: Callable[[Tensor], Tensor],
    *,
    n_w: int | None,
    reduction: PerturbationReduction = "mean",
) -> Callable[[Tensor], Tensor]:
    """Wrap an existing BoTorch sample constraint for InputPerturbation."""

    if n_w is None or int(n_w) <= 1:
        return constraint
    n_w = int(n_w)
    if getattr(constraint, "_bochan_input_perturbation_n_w", None) == n_w:
        if getattr(constraint, "_bochan_input_perturbation_reduction", None) == reduction:
            return constraint

    def wrapped(samples: Tensor) -> Tensor:
        value = constraint(samples)
        return reduce_input_perturbation_values(
            value,
            n_w=n_w,
            reduction=reduction,
        )

    setattr(wrapped, "_bochan_input_perturbation_n_w", n_w)
    setattr(wrapped, "_bochan_input_perturbation_reduction", reduction)
    setattr(wrapped, "_bochan_wrapped_constraint", constraint)
    return wrapped


def wrap_sample_constraints_for_input_perturbation(
    constraints: Sequence[Callable[[Tensor], Tensor]],
    *,
    n_w: int | None,
    reduction: PerturbationReduction = "mean",
) -> list[Callable[[Tensor], Tensor]]:
    """Wrap multiple BoTorch sample constraints for InputPerturbation."""

    return [
        wrap_sample_constraint_for_input_perturbation(
            constraint,
            n_w=n_w,
            reduction=reduction,
        )
        for constraint in constraints
    ]


def make_sample_constraint(
    spec: FeasibilityConstraintSpec | OrdinalRankConstraintSpec,
    *,
    output_names: Optional[Sequence[str]] = None,
    input_perturbation_n_w: int | None = None,
    perturbation_reduction: PerturbationReduction = "mean",
) -> Callable[[Tensor], Tensor]:
    """BoTorch MC acquisition の ``constraints`` に渡せる callable を作る。

    ``target_class`` / ``target_classes`` 付きの ``FeasibilityConstraintSpec``
    は class probability tensor を必要とするため、通常は
    ``FeasibilityWeightedAcquisition`` 経由で使う。ここでは samples が
    class probability を含む特殊ケースのみサポートする。
    """

    if isinstance(spec, OrdinalRankConstraintSpec):
        def ordinal_constraint(samples: Tensor) -> Tensor:
            value = constraint_value_from_ordinal_probs(samples, spec)
            return reduce_input_perturbation_values(
                value,
                n_w=input_perturbation_n_w,
                reduction=perturbation_reduction,
            )

        return ordinal_constraint

    idx = normalize_output_index(spec.output, output_names=output_names)

    def constraint(samples: Tensor) -> Tensor:
        if spec.has_target_classes:
            if samples.ndim >= 2 and samples.shape[-2] > idx:
                probs = samples[..., idx, :]
            else:
                probs = samples
            value = constraint_value_from_class_probs(probs, spec)
        else:
            if samples.shape[-1] <= idx:
                raise IndexError(
                    f"Constraint output index {idx} is out of range for "
                    f"samples.shape={tuple(samples.shape)}."
                )
            y = samples[..., idx]
            value = constraint_value_from_output(y, spec)
        return reduce_input_perturbation_values(
            value,
            n_w=input_perturbation_n_w,
            reduction=perturbation_reduction,
        )

    return constraint


def make_sample_constraints(
    specs: Sequence[FeasibilityConstraintSpec | OrdinalRankConstraintSpec],
    *,
    output_names: Optional[Sequence[str]] = None,
    input_perturbation_n_w: int | None = None,
    perturbation_reduction: PerturbationReduction = "mean",
) -> list[Callable[[Tensor], Tensor]]:
    """複数の constraint spec を BoTorch 互換 callable の list に変換する。"""

    return [
        make_sample_constraint(
            spec,
            output_names=output_names,
            input_perturbation_n_w=input_perturbation_n_w,
            perturbation_reduction=perturbation_reduction,
        )
        for spec in specs
    ]


def evaluate_sample_constraints(
    samples: Tensor,
    specs: Sequence[FeasibilityConstraintSpec | OrdinalRankConstraintSpec],
    *,
    output_names: Optional[Sequence[str]] = None,
    input_perturbation_n_w: int | None = None,
    perturbation_reduction: PerturbationReduction = "mean",
) -> Tensor:
    """samples 上で複数制約を評価する。"""

    values = []
    for spec in specs:
        fn = make_sample_constraint(
            spec,
            output_names=output_names,
            input_perturbation_n_w=input_perturbation_n_w,
            perturbation_reduction=perturbation_reduction,
        )
        values.append(fn(samples).unsqueeze(-1))

    if len(values) == 0:
        return torch.empty(*samples.shape[:-1], 0, device=samples.device, dtype=samples.dtype)

    return torch.cat(values, dim=-1)


def soft_feasibility_from_constraint_values(
    values: Tensor,
    *,
    eta: float = 1e-3,
    reduce_constraints: Literal["prod", "min", "mean", "none"] = "prod",
) -> Tensor:
    """constraint value から soft feasibility を計算する。"""

    if float(eta) <= 0.0:
        raise ValueError("eta must be positive.")

    pf = torch.sigmoid(-values / float(eta))

    if reduce_constraints == "none":
        return pf
    if reduce_constraints == "prod":
        return pf.prod(dim=-1)
    if reduce_constraints == "min":
        return pf.min(dim=-1).values
    if reduce_constraints == "mean":
        return pf.mean(dim=-1)

    raise ValueError("reduce_constraints must be 'prod', 'min', 'mean', or 'none'.")


__all__ = [
    "ConstraintSense",
    "FeasibilityConstraintSpec",
    "OrdinalRankConstraintSpec",
    "OrdinalRankSense",
    "OutputKey",
    "PerturbationReduction",
    "class_probability_from_probs",
    "constraint_value_from_class_probs",
    "constraint_value_from_ordinal_probs",
    "constraint_value_from_output",
    "evaluate_sample_constraints",
    "make_sample_constraint",
    "make_sample_constraints",
    "normalize_output_index",
    "ordinal_rank_probability",
    "reduce_input_perturbation_values",
    "soft_feasibility_from_constraint_values",
    "wrap_sample_constraint_for_input_perturbation",
    "wrap_sample_constraints_for_input_perturbation",
]
