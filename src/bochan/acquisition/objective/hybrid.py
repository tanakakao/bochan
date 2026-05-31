from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence, Union

import torch
from botorch.acquisition.objective import MCAcquisitionObjective
from torch import Tensor

from .regression import (
    MultiOutputRegressionInputPerturbationObjective,
    RegressionLinearMCObjective,
    RegressionScalarObjective,
    RiskType,
    _aggregate_scalar_axis,
    _validate_n_w_risk,
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


class HybridWeightedSumObjective(MCAcquisitionObjective):
    """複数の hybrid objective 出力を重み付き和で scalarize する objective。

    `qEI` / `qNEI` などの単一目的 acquisition に複数出力を渡したい場合に使う。
    内部では `RegressionLinearMCObjective` で出力選択・符号・重み・目標値変換を行い、
    最後の objective 次元を sum して scalar objective にする。
    """

    def __init__(
        self,
        inner_objective: RegressionLinearMCObjective,
        n_w: Optional[int] = None,
        risk_type: RiskType = None,
        alpha: float = 0.5,
        maximize: bool = True,
        aggregate_mean_when_no_risk: bool = True,
        allow_unexpanded: bool = True,
    ) -> None:
        super().__init__()
        self.inner_objective = inner_objective
        self.n_w = None if n_w is None else int(n_w)
        self.risk_type = risk_type
        self.alpha = float(alpha)
        self.maximize = bool(maximize)
        self.aggregate_mean_when_no_risk = bool(aggregate_mean_when_no_risk)
        self.allow_unexpanded = bool(allow_unexpanded)

        _validate_n_w_risk(
            n_w=self.n_w,
            risk_type=self.risk_type,
            alpha=self.alpha,
        )

    def _aggregate_if_needed(self, values: Tensor, X: Optional[Tensor]) -> Tensor:
        if self.n_w is None or self.n_w <= 1:
            return values
        if self.risk_type is None and not self.aggregate_mean_when_no_risk:
            return values

        n_w = int(self.n_w)

        # Baseline: X.shape = (n, d), values.shape = sample_shape x n_like
        if X is not None and X.ndim == 2:
            n = X.shape[-2]
            if values.shape[-1] == n:
                return values
            if values.ndim >= 2 and values.shape[-2] == n and values.shape[-1] == n_w:
                return _aggregate_scalar_axis(
                    values,
                    n_w=n_w,
                    risk_type=self.risk_type,
                    alpha=self.alpha,
                    risk_dim=-1,
                    maximize=self.maximize,
                )
            q_like = values.shape[-1]
            if q_like == n * n_w:
                values_w = values.reshape(*values.shape[:-1], n, n_w)
                return _aggregate_scalar_axis(
                    values_w,
                    n_w=n_w,
                    risk_type=self.risk_type,
                    alpha=self.alpha,
                    risk_dim=-1,
                    maximize=self.maximize,
                )
            if self.allow_unexpanded:
                return values
            raise RuntimeError(
                "Could not aggregate weighted-sum baseline samples. "
                f"values.shape={tuple(values.shape)}, X.shape={tuple(X.shape)}, n_w={n_w}."
            )

        # Candidate: X.shape = (*batch, q, d), values.shape = sample_shape x batch x q_like
        if X is not None and X.ndim >= 3:
            q = X.shape[-2]
            q_like = values.shape[-1]
            if q_like == q:
                return values
            if q_like == q * n_w:
                values_w = values.reshape(*values.shape[:-1], q, n_w)
                return _aggregate_scalar_axis(
                    values_w,
                    n_w=n_w,
                    risk_type=self.risk_type,
                    alpha=self.alpha,
                    risk_dim=-1,
                    maximize=self.maximize,
                )
            if self.allow_unexpanded:
                return values
            raise RuntimeError(
                "Could not aggregate weighted-sum candidate samples. "
                f"values.shape={tuple(values.shape)}, X.shape={tuple(X.shape)}, n_w={n_w}."
            )

        q_expanded = values.shape[-1]
        if q_expanded % n_w != 0:
            if self.allow_unexpanded:
                return values
            raise RuntimeError(
                "values.shape[-1] must be divisible by n_w for InputPerturbation aggregation. "
                f"Got values.shape={tuple(values.shape)}, n_w={n_w}."
            )
        q = q_expanded // n_w
        values_w = values.reshape(*values.shape[:-1], q, n_w)
        return _aggregate_scalar_axis(
            values_w,
            n_w=n_w,
            risk_type=self.risk_type,
            alpha=self.alpha,
            risk_dim=-1,
            maximize=self.maximize,
        )

    def forward(self, samples: Tensor, X: Optional[Tensor] = None) -> Tensor:
        values = self.inner_objective(samples=samples, X=X)
        if values.ndim < 1:
            raise RuntimeError("inner_objective returned a scalar; expected at least one dimension.")
        scalar = values.sum(dim=-1)
        return self._aggregate_if_needed(scalar, X=X)


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
    """Hybrid / non-hybrid 共通の multi-output linear objective factory。

    Notes:
        返り値は multi-output objective なので、qEHVI / qNEHVI などの多目的 acquisition 向け。
        qEI / qNEI などの単一目的 acquisition には `make_hybrid_weighted_sum_objective` を使う。
    """

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


def make_hybrid_weighted_sum_objective(
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
) -> HybridWeightedSumObjective:
    """複数出力を重み付き和で単一目的にする factory。

    qEI / qNEI / qUCB など、scalar objective を期待する acquisition に使う。
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
    return HybridWeightedSumObjective(
        inner_objective=inner,
        n_w=n_w,
        risk_type=risk_type,
        alpha=alpha,
        maximize=maximize,
        aggregate_mean_when_no_risk=aggregate_mean_when_no_risk,
        allow_unexpanded=allow_unexpanded,
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

    Notes:
        返り値は multi-output objective なので、qEHVI / qNEHVI などの多目的 acquisition 向け。
        qEI / qNEI などの単一目的 acquisition には `make_hybrid_weighted_sum_objective` を使う。

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
    "HybridWeightedSumObjective",
    "OutputKey",
    "make_hybrid_linear_objective",
    "make_hybrid_multi_output_objective",
    "make_hybrid_objective_specs",
    "make_hybrid_scalar_objective",
    "make_hybrid_weighted_sum_objective",
    "resolve_hybrid_output_indices",
]
