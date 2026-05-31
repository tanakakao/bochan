from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence, Union

import torch
from torch import Tensor

from .regression import (
    MultiOutputRegressionInputPerturbationObjective,
    RegressionLinearMCObjective,
    RegressionScalarObjective,
    RiskType,
)

OutputKey = Union[int, str]
Direction = Literal["maximize", "minimize"]


@dataclass(frozen=True)
class HybridObjectiveSpec:
    """Hybrid / non-hybrid multi-output model 共通の objective 設定。

    `OutputSpec` は「出力が何か」を表し、`HybridObjectiveSpec` は
    「今回の最適化でその出力をどう使うか」を表す。

    Args:
        output:
            出力名または出力 index。
        direction:
            目的方向。`maximize` なら +1、`minimize` なら -1 に変換する。
        weight:
            scalarization / multi-output objective 用の重み。
        eq_target:
            目標値に近いほどよい目的にする場合の目標値。
    """

    output: OutputKey
    direction: Direction = "maximize"
    weight: float = 1.0
    eq_target: Optional[float] = None

    def __post_init__(self) -> None:
        if self.direction not in {"maximize", "minimize"}:
            raise ValueError("direction must be 'maximize' or 'minimize'.")
        if float(self.weight) < 0.0:
            raise ValueError("weight must be non-negative.")

    @property
    def sign(self) -> float:
        return 1.0 if self.direction == "maximize" else -1.0


def _get_output_names(model) -> Optional[list[str]]:
    names = getattr(model, "output_names", None)
    if names is None:
        return None
    if callable(names):
        names = names()
    return list(names)


def resolve_hybrid_output_indices(
    model,
    outputs: Optional[OutputKey | Sequence[OutputKey] | Tensor] = None,
) -> list[int]:
    """出力名 / index を整数 index に変換する。

    `HybridMultiOutputModel` では `model.output_names` を使う。
    通常の multi-output model では文字列指定は使えないため、整数 index を使う。
    """

    if outputs is None:
        num_outputs = getattr(model, "num_outputs", None)
        if num_outputs is None:
            raise ValueError("outputs=None requires model.num_outputs.")
        return list(range(int(num_outputs)))

    if torch.is_tensor(outputs):
        outputs = outputs.detach().cpu().tolist()
    if isinstance(outputs, (int, str)):
        outputs = [outputs]

    output_names = _get_output_names(model)
    name_to_idx = None if output_names is None else {name: i for i, name in enumerate(output_names)}

    indices: list[int] = []
    for output in outputs:
        if isinstance(output, str):
            if name_to_idx is None:
                raise ValueError(
                    "String output names require model.output_names. "
                    "Use integer indices for non-hybrid multi-output models."
                )
            if output not in name_to_idx:
                raise KeyError(f"Unknown output name {output!r}. Available={output_names}.")
            idx = name_to_idx[output]
        else:
            idx = int(output)

        if idx < 0:
            raise IndexError(f"output index must be non-negative. Got {idx}.")
        indices.append(idx)

    return indices


def _as_list_or_default(
    values,
    *,
    n: int,
    default,
    name: str,
) -> list:
    if values is None:
        return [default] * n
    if isinstance(values, (str, int, float)):
        values = [values]
    values = list(values)
    if len(values) != n:
        raise ValueError(f"{name} must have length {n}. Got {len(values)}.")
    return values


def _direction_to_sign(direction: Direction | bool | float | int) -> float:
    if isinstance(direction, str):
        if direction == "maximize":
            return 1.0
        if direction == "minimize":
            return -1.0
        raise ValueError("direction must be 'maximize' or 'minimize'.")
    if isinstance(direction, bool):
        return 1.0 if direction else -1.0
    sign = float(direction)
    if sign == 0.0:
        raise ValueError("sign must be non-zero.")
    return 1.0 if sign > 0.0 else -1.0


def make_hybrid_objective_specs(
    outputs: Sequence[OutputKey],
    *,
    directions: Optional[Sequence[Direction | bool | float | int]] = None,
    weights: Optional[Sequence[float]] = None,
    eq_targets: Optional[Sequence[Optional[float]]] = None,
) -> list[HybridObjectiveSpec]:
    """UI / API の配列設定から `HybridObjectiveSpec` を作る。"""

    outputs_l = list(outputs)
    n = len(outputs_l)
    directions_l = _as_list_or_default(directions, n=n, default="maximize", name="directions")
    weights_l = _as_list_or_default(weights, n=n, default=1.0, name="weights")
    eq_targets_l = _as_list_or_default(eq_targets, n=n, default=None, name="eq_targets")

    specs = []
    for output, direction, weight, eq_target in zip(outputs_l, directions_l, weights_l, eq_targets_l):
        if isinstance(direction, str):
            direction_s: Direction = direction  # type: ignore[assignment]
        else:
            direction_s = "maximize" if _direction_to_sign(direction) > 0.0 else "minimize"
        specs.append(
            HybridObjectiveSpec(
                output=output,
                direction=direction_s,
                weight=float(weight),
                eq_target=None if eq_target is None else float(eq_target),
            )
        )
    return specs


def _normalize_objective_specs(
    specs: Optional[Sequence[HybridObjectiveSpec]],
    *,
    outputs: Optional[OutputKey | Sequence[OutputKey] | Tensor],
    directions: Optional[Sequence[Direction | bool | float | int]],
    weights: Optional[Sequence[float]],
    eq_targets: Optional[Sequence[Optional[float]]],
    model,
) -> list[HybridObjectiveSpec]:
    if specs is not None:
        return list(specs)

    if outputs is None:
        outputs = resolve_hybrid_output_indices(model, None)
    elif torch.is_tensor(outputs):
        outputs = outputs.detach().cpu().tolist()
    elif isinstance(outputs, (int, str)):
        outputs = [outputs]

    return make_hybrid_objective_specs(
        list(outputs),
        directions=directions,
        weights=weights,
        eq_targets=eq_targets,
    )


def make_hybrid_scalar_objective(
    model,
    output: OutputKey,
    *,
    direction: Direction | bool | float | int = "maximize",
    weight: float = 1.0,
    eq_target: Optional[float] = None,
    n_w: Optional[int] = None,
    risk_type: RiskType = None,
    alpha: float = 0.5,
    maximize: bool = True,
    aggregate_mean_when_no_risk: bool = True,
    allow_unexpanded: bool = True,
) -> RegressionScalarObjective:
    """Hybrid / non-hybrid 共通の single-output objective factory。

    Notes:
        `OutputSpec.sign` / `OutputSpec.weight` には依存しない。
        方向と重みはこの objective factory の引数で指定する。
    """

    idx = resolve_hybrid_output_indices(model, [output])[0]
    sign = _direction_to_sign(direction)

    return RegressionScalarObjective(
        output_index=idx,
        weight=float(weight),
        sign=sign,
        eq_target=eq_target,
        n_w=n_w,
        risk_type=risk_type,
        alpha=alpha,
        maximize=maximize,
        aggregate_mean_when_no_risk=aggregate_mean_when_no_risk,
        allow_unexpanded=allow_unexpanded,
    )


def make_hybrid_linear_objective(
    model,
    specs: Optional[Sequence[HybridObjectiveSpec]] = None,
    *,
    outputs: Optional[OutputKey | Sequence[OutputKey] | Tensor] = None,
    directions: Optional[Sequence[Direction | bool | float | int]] = None,
    weights: Optional[Sequence[float]] = None,
    eq_targets: Optional[Sequence[Optional[float]]] = None,
    dtype: torch.dtype = torch.double,
    device: Optional[torch.device] = None,
) -> RegressionLinearMCObjective:
    """Hybrid / non-hybrid 共通の multi-output linear objective factory。"""

    objective_specs = _normalize_objective_specs(
        specs,
        outputs=outputs,
        directions=directions,
        weights=weights,
        eq_targets=eq_targets,
        model=model,
    )
    if len(objective_specs) == 0:
        raise ValueError("At least one objective output is required.")

    output_indices = resolve_hybrid_output_indices(model, [s.output for s in objective_specs])
    signs = [s.sign for s in objective_specs]
    weights_l = [float(s.weight) for s in objective_specs]
    eq_targets_l = [s.eq_target for s in objective_specs]

    return RegressionLinearMCObjective(
        output_indices=output_indices,
        weights=weights_l,
        signs=signs,
        eq_targets=eq_targets_l,
        dtype=dtype,
        device=device,
    )


def make_hybrid_multi_output_objective(
    model,
    specs: Optional[Sequence[HybridObjectiveSpec]] = None,
    *,
    outputs: Optional[OutputKey | Sequence[OutputKey] | Tensor] = None,
    directions: Optional[Sequence[Direction | bool | float | int]] = None,
    weights: Optional[Sequence[float]] = None,
    eq_targets: Optional[Sequence[Optional[float]]] = None,
    n_w: Optional[int] = None,
    risk_type: RiskType = None,
    alpha: float = 0.5,
    maximize: bool = True,
    aggregate_mean_when_no_risk: bool = True,
    allow_unexpanded: bool = True,
    dtype: torch.dtype = torch.double,
    device: Optional[torch.device] = None,
):
    """Hybrid / non-hybrid 共通の multi-output objective factory。

    `n_w` が None または 1 の場合は `RegressionLinearMCObjective` を返す。
    `n_w > 1` の場合は `MultiOutputRegressionInputPerturbationObjective` で包む。
    """

    inner = make_hybrid_linear_objective(
        model=model,
        specs=specs,
        outputs=outputs,
        directions=directions,
        weights=weights,
        eq_targets=eq_targets,
        dtype=dtype,
        device=device,
    )

    if n_w is None or int(n_w) <= 1:
        return inner

    return MultiOutputRegressionInputPerturbationObjective(
        inner_objective=inner,
        n_w=int(n_w),
        risk_type=risk_type,
        alpha=alpha,
        maximize=maximize,
        aggregate_mean_when_no_risk=aggregate_mean_when_no_risk,
        allow_unexpanded=allow_unexpanded,
    )


__all__ = [
    "Direction",
    "HybridObjectiveSpec",
    "OutputKey",
    "make_hybrid_linear_objective",
    "make_hybrid_multi_output_objective",
    "make_hybrid_objective_specs",
    "make_hybrid_scalar_objective",
    "resolve_hybrid_output_indices",
]
