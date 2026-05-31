from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional, Sequence, Union

import torch
from torch import Tensor


ConstraintSense = Literal["ge", "le", "eq"]
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


def make_sample_constraint(
    spec: FeasibilityConstraintSpec,
    *,
    output_names: Optional[Sequence[str]] = None,
) -> Callable[[Tensor], Tensor]:
    """BoTorch MC acquisition の ``constraints`` に渡せる callable を作る。

    Args:
        spec:
            制約定義。
        output_names:
            ``spec.output`` が文字列の場合に参照する出力名リスト。

    Returns:
        Callable:
            ``samples -> constraint_value``。
            feasible な sample では ``constraint_value <= 0`` になる。
    """

    idx = normalize_output_index(spec.output, output_names=output_names)

    def constraint(samples: Tensor) -> Tensor:
        if samples.shape[-1] <= idx:
            raise IndexError(
                f"Constraint output index {idx} is out of range for "
                f"samples.shape={tuple(samples.shape)}."
            )
        y = samples[..., idx]
        return constraint_value_from_output(y, spec)

    return constraint


def make_sample_constraints(
    specs: Sequence[FeasibilityConstraintSpec],
    *,
    output_names: Optional[Sequence[str]] = None,
) -> list[Callable[[Tensor], Tensor]]:
    """複数の constraint spec を BoTorch 互換 callable の list に変換する。"""

    return [make_sample_constraint(spec, output_names=output_names) for spec in specs]


def evaluate_sample_constraints(
    samples: Tensor,
    specs: Sequence[FeasibilityConstraintSpec],
    *,
    output_names: Optional[Sequence[str]] = None,
) -> Tensor:
    """samples 上で複数制約を評価する。

    Returns:
        Tensor:
            shape は ``samples.shape[:-1] + (n_constraints,)``。
            最後の制約次元の各値が ``<= 0`` なら feasible。
    """

    values = []
    for spec in specs:
        fn = make_sample_constraint(spec, output_names=output_names)
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
    "OutputKey",
    "constraint_value_from_output",
    "evaluate_sample_constraints",
    "make_sample_constraint",
    "make_sample_constraints",
    "normalize_output_index",
    "soft_feasibility_from_constraint_values",
]
