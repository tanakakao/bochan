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

    BoTorch の MC acquisition に渡す constraint callable は、
    feasible なときに ``constraint(samples) <= 0`` を返す必要がある。

    Args:
        output:
            posterior samples の最後の出力次元 index、または
            `HybridMultiOutputModel.output_names` に含まれる出力名。
        threshold:
            制約の閾値。
        sense:
            - ``"ge"``: ``y >= threshold`` を feasible とする。
            - ``"le"``: ``y <= threshold`` を feasible とする。
            - ``"eq"``: ``abs(y - threshold) <= margin`` を feasible とする。
        margin:
            ``sense="eq"`` の許容幅。
        scale:
            constraint value のスケール調整。BoTorch の sigmoid feasibility で
            constraint 間の感度を揃えたい場合に使う。
    """

    output: OutputKey
    threshold: float
    sense: ConstraintSense = "ge"
    margin: float = 0.0
    scale: float = 1.0

    def __post_init__(self) -> None:
        if self.sense not in {"ge", "le", "eq"}:
            raise ValueError("sense must be one of 'ge', 'le', or 'eq'.")
        if float(self.scale) <= 0.0:
            raise ValueError("scale must be positive.")
        if self.sense == "eq" and float(self.margin) < 0.0:
            raise ValueError("margin must be non-negative for equality constraints.")


@dataclass(frozen=True)
class OrdinalRankConstraintSpec:
    """Ordinal rank probability を使う feasible constraint 定義。

    例:
        ``OrdinalRankConstraintSpec("quality_rank", rank=2, sense="ge", probability_threshold=0.8)``
        は ``P(y >= 2) >= 0.8`` を feasible とする。

    Args:
        output:
            ordinal 出力の index または name。
            `FeasibilityWeightedAcquisition` では `HybridMultiOutputModel.output_names` と
            `class_probs_list` を使って該当 ordinal 出力の class probability を取得する。
        rank:
            しきい値にするランク。通常は 0 始まりのクラス index。
        sense:
            - ``"ge"``: ``P(y >= rank) >= probability_threshold``。
            - ``"le"``: ``P(y <= rank) >= probability_threshold``。
            - ``"eq"``: ``P(y == rank) >= probability_threshold``。
        probability_threshold:
            feasibility に必要な累積 / 点確率の下限。
        scale:
            constraint value のスケール調整。
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
        # y >= threshold  <=>  threshold - y <= 0
        value = threshold - y
    elif spec.sense == "le":
        # y <= threshold  <=>  y - threshold <= 0
        value = y - threshold
    elif spec.sense == "eq":
        # |y - threshold| <= margin  <=>  |y - threshold| - margin <= 0
        value = torch.abs(y - threshold) - float(spec.margin)
    else:
        raise ValueError(f"Unknown sense={spec.sense!r}.")

    return value / scale


def ordinal_rank_probability(probs: Tensor, spec: OrdinalRankConstraintSpec) -> Tensor:
    """ordinal class probabilities から rank 条件の確率を計算する。

    Args:
        probs:
            shape ``[..., K]`` の class probability。
        spec:
            ordinal rank constraint spec。

    Returns:
        Tensor:
            shape ``probs.shape[:-1]``。
    """

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
    """ordinal probabilities から BoTorch 形式の constraint value を計算する。

    feasible 条件は ``ordinal_rank_probability(probs, spec) >= probability_threshold``。
    戻り値は feasible なときに ``<= 0`` になる。
    """

    p_rank = ordinal_rank_probability(probs, spec)
    value = float(spec.probability_threshold) - p_rank
    return value / float(spec.scale)


def reduce_input_perturbation_values(
    value: Tensor,
    *,
    n_w: int | None,
    reduction: PerturbationReduction = "mean",
) -> Tensor:
    """Reduce the perturbation-expanded q dimension back to the original q.

    InputPerturbation expands an input q-batch from ``q`` to ``q * n_w``. BoTorch
    constrained MC acquisitions multiply acquisition values shaped ``... x q``
    by feasibility indicators from the raw posterior samples. Without reducing
    the constraint value, the indicator keeps shape ``... x (q * n_w)`` and can
    no longer be multiplied with the acquisition values.
    """

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
    """Wrap an existing BoTorch sample constraint for InputPerturbation.

    The wrapped callable first evaluates the original constraint, then reduces
    the final q-like dimension from ``q * n_w`` back to ``q``.
    """

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

    Notes:
        ``FeasibilityConstraintSpec`` は samples の最後の出力次元から1列を取り出す。
        ``OrdinalRankConstraintSpec`` は、samples の最後の次元に ordinal class probability
        が並んでいる場合に使う。つまり samples 自体が ``[..., K]``、または
        ``output`` で選んだ要素が ``[..., K]`` になる posterior を想定する。
        通常の `HybridMultiOutputModel.objective_posterior` は expected utility だけを
        返すため、BoTorch 標準 ``constraints=`` では `FeasibilityConstraintSpec` を使う。
        rank probability 制約は `FeasibilityWeightedAcquisition` と組み合わせるのが基本。
    """

    if isinstance(spec, OrdinalRankConstraintSpec):
        # samples が [..., K] の ordinal probability そのものとして渡るケースを扱う。
        # output が指定されている場合でも、通常の [..., m] 形式から K クラス確率を
        # 復元することはできないため、ここでは最後の次元を class dimension とみなす。
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
    """samples 上で複数制約を評価する。

    Returns:
        Tensor:
            shape は ``samples.shape[:-1] + (n_constraints,)``。
            最後の制約次元の各値が ``<= 0`` なら feasible。
    """

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
    """constraint value から soft feasibility を計算する。

    Args:
        values:
            ``[..., n_constraints]``。各 constraint は ``<= 0`` が feasible。
        eta:
            sigmoid の温度。小さいほど hard constraint に近い。
        reduce_constraints:
            複数制約の集約方法。

    Returns:
        Tensor:
            ``reduce_constraints='none'`` の場合は ``values`` と同じ shape。
            それ以外は最後の制約次元を集約した Tensor。
    """

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
