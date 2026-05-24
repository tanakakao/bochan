from __future__ import annotations

import math
from typing import Callable, List, Literal, Optional, Sequence

import torch
from torch import Tensor

from botorch.acquisition.objective import MCAcquisitionObjective
from botorch.acquisition.multi_objective.objective import MCMultiOutputObjective


RiskType = Optional[Literal["var", "cvar"]]


# ============================================================
# Common helpers
# ============================================================


def _validate_same_length(**items: Sequence[object]) -> None:
    lengths = {name: len(value) for name, value in items.items()}

    if len(set(lengths.values())) != 1:
        raise ValueError(f"All inputs must have the same length. Got: {lengths}")


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
    if risk_type is None:
        return values_w.mean(dim=risk_dim)

    # maximize=True: smaller values are worse.
    # maximize=False: larger values are worse.
    descending = not maximize
    sorted_values = torch.sort(values_w, dim=risk_dim, descending=descending).values

    k = max(1, int(math.ceil(int(n_w) * float(alpha))))
    tail = sorted_values.narrow(dim=risk_dim, start=0, length=k)

    if risk_type == "var":
        return tail.select(dim=risk_dim, index=k - 1)

    if risk_type == "cvar":
        return tail.mean(dim=risk_dim)

    raise ValueError(f"Unknown risk_type: {risk_type!r}.")


def _aggregate_multioutput_axis(
    values_w: Tensor,
    *,
    n_w: int,
    risk_type: RiskType,
    alpha: float,
    risk_dim: int = -2,
    maximize: bool = True,
) -> Tensor:
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


# ============================================================
# 1. Single-output regression objective
#    posterior samples -> scalar objective
# ============================================================


class RegressionScalarObjective(MCAcquisitionObjective):
    """regression 用 objective。posterior samples または点ごとの score を scalar value に変換します。

    Notes:
        regression active learning の獲得関数では、posterior samples ではなく
        ``score.shape = batch_shape x q`` のような点ごとの score を objective に渡す場合がある。
        このとき最後の次元は output_dim ではなく q-batch 次元なので、
        ``samples[..., output_index]`` で q 次元を落としてはいけない。

        一方、通常の MC acquisition では
        ``samples.shape = sample_shape x batch_shape x q x m`` の最後の次元 m が
        output_dim なので、従来どおり output_index で scalarize する。
    
    Args:
        output_index: single-output へ scalarize するときに使う出力列 index。
        weight: score または objective に掛ける重み。
        sign: 目的の向きを揃える符号。最大化なら +1、最小化なら -1 を使います。
        eq_target: 目標値に近いほど良い score に変換する場合の目標値。
        n_w: InputPerturbation で 1 点あたりに展開される摂動数。
        risk_type: InputPerturbation 集約の risk 種類。`None`、`var`、`cvar`。
        alpha: risk 集約または qNEHVI の近似設定に使うパラメータ。
        maximize: score が大きいほど良い向きに揃っているかどうか。
        aggregate_mean_when_no_risk: risk_type=None のときも摂動方向を平均集約するか。
        allow_unexpanded: shape から摂動展開を判定できない場合にそのまま返すか。
    
    Returns:
        Tensor: 入力 samples または score を変換・集約した objective value。
    """

    def __init__(
        self,
        output_index: int = 0,
        weight: float = 1.0,
        sign: float = 1.0,
        eq_target: Optional[float] = None,
        n_w: Optional[int] = None,
        risk_type: RiskType = None,
        alpha: float = 0.5,
        maximize: bool = True,
        aggregate_mean_when_no_risk: bool = True,
        allow_unexpanded: bool = True,
    ) -> None:
        super().__init__()

        self.output_index = int(output_index)
        self.weight = float(weight)
        self.sign = float(sign)
        self.eq_target = None if eq_target is None else float(eq_target)
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

    def _looks_like_pointwise_score(self, samples: Tensor, X: Optional[Tensor]) -> bool:
        """Return True when samples already has pointwise score shape (..., q_like).

        For active-learning acquisitions, score is often passed as ``batch_shape x q``.
        BoTorch's ``MCAcquisitionObjective.__call__`` then verifies that the final
        dimension agrees with ``X.shape[-2]``. Treating this final dimension as an
        output dimension would incorrectly collapse q=1 to ``batch_shape`` and
        produce the error ``Got batch_size and 1``.
        """
        if X is None or samples.ndim == 0:
            return False

        # Candidate case: X is batch_shape x q x d and score is batch_shape x q
        # or batch_shape x (q * n_w). This is the failing path for qRegressionBALD
        # during optimize_acqf initial-condition evaluation.
        if X.ndim >= 3 and samples.ndim == X.ndim - 1:
            q = X.shape[-2]
            q_like = samples.shape[-1]
            if q_like == q:
                return True
            if self.n_w is not None and q_like == q * int(self.n_w):
                return True

        # Baseline / unbatched case: X is n x d and score is n or n*n_w.
        if X.ndim == 2 and samples.ndim == 1:
            n = X.shape[-2]
            n_like = samples.shape[-1]
            if n_like == n:
                return True
            if self.n_w is not None and n_like == n * int(self.n_w):
                return True

        return False

    def _scalarize(self, samples: Tensor, X: Optional[Tensor] = None) -> Tensor:
        if samples.ndim == 0:
            raise RuntimeError("samples must have at least one dimension.")

        if self._looks_like_pointwise_score(samples, X):
            y = samples
        elif samples.ndim == 1:
            if self.output_index != 0:
                raise IndexError(
                    f"samples is 1D, but output_index={self.output_index}."
                )
            y = samples
        else:
            if self.output_index >= samples.shape[-1]:
                raise IndexError(
                    f"output_index={self.output_index} is out of bounds "
                    f"for samples.shape[-1]={samples.shape[-1]}."
                )
            y = samples[..., self.output_index]

        if self.eq_target is not None:
            return -torch.abs(y - self.eq_target) * self.weight

        return y * self.sign * self.weight

    def forward(self, samples: Tensor, X: Optional[Tensor] = None) -> Tensor:
        if not torch.is_tensor(samples):
            raise TypeError(f"samples must be a Tensor. Got {type(samples)}.")

        values = self._scalarize(samples, X=X)

        if self.n_w is None or self.n_w <= 1:
            return values

        if self.risk_type is None and not self.aggregate_mean_when_no_risk:
            return values

        n_w = int(self.n_w)

        # Baseline: X.shape = (n, d)
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
                "Could not aggregate regression baseline samples. "
                f"values.shape={tuple(values.shape)}, "
                f"X.shape={tuple(X.shape)}, n_w={n_w}."
            )

        # Candidate: X.shape = (*batch, q, d)
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
                "Could not aggregate regression candidate samples. "
                f"values.shape={tuple(values.shape)}, "
                f"X.shape={tuple(X.shape)}, n_w={n_w}."
            )

        # X is None fallback
        q_expanded = values.shape[-1]

        if q_expanded % n_w != 0:
            if self.allow_unexpanded:
                return values

            raise RuntimeError(
                "values.shape[-1] must be divisible by n_w for "
                "InputPerturbation aggregation. "
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


# ============================================================
# 2. Multi-output regression linear objective
#    posterior samples -> selected multi-output values
# ============================================================


class RegressionLinearMCObjective(MCMultiOutputObjective):
    """regression 用 objective。posterior samples を scalar または multi-output objective に変換します。
    
    Args:
        output_indices: multi-output objective に残す出力列 index のリスト。
        weights: NParEGO や weighted objective で使う scalarization weight。
        signs: 各出力を最大化方向に揃える符号。
        eq_targets: 各出力の目標値。None の出力は通常の線形変換を使います。
        dtype: この acquisition / objective の動作を制御するパラメータ。
        device: この acquisition / objective の動作を制御するパラメータ。
    
    Returns:
        Tensor: 入力 samples または score を変換・集約した objective value。
    """

    def __init__(
        self,
        output_indices: Sequence[int],
        weights: Sequence[float] | Tensor,
        signs: Sequence[float] | Tensor,
        eq_targets: Optional[Sequence[Optional[float]]] = None,
        *,
        dtype: torch.dtype = torch.double,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()

        if eq_targets is None:
            eq_targets = [None] * len(output_indices)

        _validate_same_length(
            output_indices=output_indices,
            weights=weights,
            signs=signs,
            eq_targets=eq_targets,
        )

        output_indices_tensor = torch.as_tensor(
            output_indices,
            dtype=torch.long,
            device=device,
        )
        weights_tensor = torch.as_tensor(
            weights,
            dtype=dtype,
            device=device,
        )
        signs_tensor = torch.as_tensor(
            signs,
            dtype=dtype,
            device=device,
        )

        eq_mask = torch.tensor(
            [target is not None for target in eq_targets],
            device=device,
        )
        eq_values = torch.tensor(
            [
                float(target) if target is not None else float("nan")
                for target in eq_targets
            ],
            dtype=dtype,
            device=device,
        )

        self.register_buffer("output_indices", output_indices_tensor)
        self.register_buffer("weights", weights_tensor)
        self.register_buffer("signs", signs_tensor)
        self.register_buffer("eq_mask", eq_mask)
        self.register_buffer("eq_values", eq_values)

    @property
    def constraints_idx(self) -> List[int]:
        """Backward-compatible list form of selected output indices."""
        return self.output_indices.detach().cpu().tolist()

    def forward(self, samples: Tensor, X: Optional[Tensor] = None) -> Tensor:
        if samples.ndim < 1:
            raise ValueError("samples must have at least one dimension.")

        if self.output_indices.numel() == 0:
            raise ValueError("At least one output index is required.")

        if int(self.output_indices.max()) >= samples.shape[-1]:
            raise IndexError(
                f"output index {int(self.output_indices.max())} is out of bounds "
                f"for samples with last dimension {samples.shape[-1]}."
            )

        idx = self.output_indices.to(device=samples.device)
        selected = samples.index_select(dim=-1, index=idx)

        weights = self.weights.to(device=samples.device, dtype=samples.dtype)
        signs = self.signs.to(device=samples.device, dtype=samples.dtype)
        eq_mask = self.eq_mask.to(device=samples.device)
        eq_values = self.eq_values.to(device=samples.device, dtype=samples.dtype)

        values = selected * signs * weights

        if bool(eq_mask.any()):
            target_values = (
                -torch.abs(selected[..., eq_mask] - eq_values[eq_mask])
                * weights[eq_mask]
            )
            values = values.clone()
            values[..., eq_mask] = target_values

        return values



# ============================================================
# 3. Multi-output regression input perturbation objective
#    posterior samples -> transformed objective -> q / n aggregation
# ============================================================


class MultiOutputRegressionInputPerturbationObjective(MCMultiOutputObjective):
    """multi-output regression 用 objective。posterior samples を scalar または multi-output objective に変換します。
    
    Args:
        inner_objective: InputPerturbation 集約前に posterior samples に適用する inner objective。
        n_w: InputPerturbation で 1 点あたりに展開される摂動数。
        risk_type: InputPerturbation 集約の risk 種類。`None`、`var`、`cvar`。
        alpha: risk 集約または qNEHVI の近似設定に使うパラメータ。
        maximize: score が大きいほど良い向きに揃っているかどうか。
        aggregate_mean_when_no_risk: この acquisition / objective の動作を制御するパラメータ。
        allow_unexpanded: この acquisition / objective の動作を制御するパラメータ。
    
    Returns:
        Tensor: 入力 samples または score を変換・集約した objective value。
    """

    def __init__(
        self,
        inner_objective: MCMultiOutputObjective,
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

    def _aggregate_risk_axis(self, values_w: Tensor) -> Tensor:
        return _aggregate_multioutput_axis(
            values_w,
            n_w=int(self.n_w),
            risk_type=self.risk_type,
            alpha=self.alpha,
            risk_dim=-2,
            maximize=self.maximize,
        )

    def forward(self, samples: Tensor, X: Optional[Tensor] = None) -> Tensor:
        if not torch.is_tensor(samples):
            raise TypeError(f"samples must be a Tensor. Got {type(samples)}.")

        values = self.inner_objective(samples=samples, X=X)

        if values.ndim < 2:
            raise RuntimeError(
                "inner_objective must return at least shape (..., q_like, m). "
                f"Got values.shape={tuple(values.shape)}."
            )

        if self.n_w is None or self.n_w <= 1:
            return values

        if self.risk_type is None and not self.aggregate_mean_when_no_risk:
            return values

        n_w = int(self.n_w)
        m = values.shape[-1]

        # Baseline: X.shape = (n, d)
        if X is not None and X.ndim == 2:
            n = X.shape[-2]

            if values.shape[-2] == n:
                return values

            if values.ndim >= 3 and values.shape[-3] == n and values.shape[-2] == n_w:
                return self._aggregate_risk_axis(values)

            q_like = values.shape[-2]

            if q_like == n * n_w:
                values_w = values.reshape(*values.shape[:-2], n, n_w, m)
                return self._aggregate_risk_axis(values_w)

            if self.allow_unexpanded:
                return values

            raise RuntimeError(
                "Could not aggregate regression baseline samples. "
                f"values.shape={tuple(values.shape)}, "
                f"X.shape={tuple(X.shape)}, n_w={n_w}."
            )

        # Candidate: X.shape = (*batch, q, d)
        if X is not None and X.ndim >= 3:
            q = X.shape[-2]
            q_like = values.shape[-2]

            if q_like == q:
                return values

            if q_like == q * n_w:
                values_w = values.reshape(*values.shape[:-2], q, n_w, m)
                return self._aggregate_risk_axis(values_w)

            if self.allow_unexpanded:
                return values

            raise RuntimeError(
                "Could not aggregate regression candidate samples. "
                f"values.shape={tuple(values.shape)}, "
                f"X.shape={tuple(X.shape)}, n_w={n_w}."
            )

        # X is None fallback
        q_expanded = values.shape[-2]

        if q_expanded % n_w != 0:
            if self.allow_unexpanded:
                return values

            raise RuntimeError(
                "values.shape[-2] must be divisible by n_w for "
                "InputPerturbation aggregation. "
                f"Got values.shape={tuple(values.shape)}, n_w={n_w}."
            )

        q = q_expanded // n_w
        values_w = values.reshape(*values.shape[:-2], q, n_w, m)

        return self._aggregate_risk_axis(values_w)



# ============================================================
# 4. Optional callable helper
# ============================================================


def make_regression_scalar_callable(
    output_index: int,
    weight: float = 1.0,
    sign: float = 1.0,
    eq_target: Optional[float] = None,
) -> Callable[[Tensor, Optional[Tensor]], Tensor]:
    """
    Create a lightweight scalar callable.

    This helper is kept for cases where BoTorch's GenericMCObjective is preferred.
    The main class-based API is RegressionScalarObjective.
    """

    idx = int(output_index)
    weight_f = float(weight)
    sign_f = float(sign)
    target = None if eq_target is None else float(eq_target)

    def scalar_obj(samples: Tensor, X: Optional[Tensor] = None) -> Tensor:
        if samples.ndim == 1:
            if idx != 0:
                raise ValueError(f"samples is 1D but output_index={idx}.")
            y = samples
        else:
            if idx >= samples.shape[-1]:
                raise IndexError(
                    f"output_index={idx} is out of bounds for "
                    f"samples.shape[-1]={samples.shape[-1]}."
                )
            y = samples[..., idx]

        if target is not None:
            return -torch.abs(y - target) * weight_f

        return y * sign_f * weight_f

    return scalar_obj

__all__ = [
    "RegressionScalarObjective",
    "RegressionLinearMCObjective",
    "MultiOutputRegressionInputPerturbationObjective",
    "make_regression_scalar_callable",
]
