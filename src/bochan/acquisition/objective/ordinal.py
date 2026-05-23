from __future__ import annotations

import math
from typing import Callable, Literal, Optional, Sequence

import torch
from torch import Tensor
from torch import nn

from botorch.acquisition.objective import MCAcquisitionObjective
from botorch.acquisition.multi_objective.objective import MCMultiOutputObjective


RiskType = Optional[Literal["var", "cvar"]]
AggregatedRiskMode = Literal["ignore", "error"]

OrdinalScoreShapeMode = Literal[
    "auto",
    "pointwise",       # (*batch, q_like)
    "multioutput_qm",  # (*batch, q_like, m)
    "multioutput_mq",  # (*batch, m, q_like)
    "aggregated",      # (*batch,)
]


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


def _canonicalize_utility_values(
    utility_values: Sequence[float] | Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    utilities = torch.as_tensor(
        utility_values,
        device=device,
        dtype=dtype,
    )

    if utilities.ndim != 1:
        raise ValueError(
            "utility_values must be 1D. "
            f"Got shape={tuple(utilities.shape)}."
        )

    return utilities


def _get_cutpoints_from_ordinal_likelihood(likelihood) -> Tensor:
    """
    OrdinalLogitLikelihood から cutpoints を取得する。

    あなたの OrdinalLogitLikelihood の属性名に合わせて、
    必要なら candidate_names に追加してください。
    """

    candidate_names = [
        "cutpoints",
        "thresholds",
        "cuts",
        "boundaries",
        "_cutpoints",
    ]

    for name in candidate_names:
        if hasattr(likelihood, name):
            value = getattr(likelihood, name)

            if callable(value):
                value = value()

            if torch.is_tensor(value):
                return value

    raise AttributeError(
        "Could not find cutpoints in ordinal likelihood. "
        "Expected one of: cutpoints, thresholds, cuts, boundaries, _cutpoints."
    )


def ordinal_logit_probs_from_latent(
    latent_f: Tensor,
    cutpoints: Tensor,
    eps: float = 1e-12,
) -> Tensor:
    """
    latent f samples を ordinal class probability に変換する。

    Ordinal logit:
        P(y <= k | f) = sigmoid(c_k - f)

    Args:
        latent_f:
            shape:
                (..., q_like)
                or (..., q_like, 1)

        cutpoints:
            shape:
                [K - 1]

    Returns:
        probs:
            shape:
                (..., q_like, K)
    """
    if latent_f.ndim >= 1 and latent_f.shape[-1] == 1:
        latent_f = latent_f.squeeze(-1)

    cutpoints = cutpoints.to(
        device=latent_f.device,
        dtype=latent_f.dtype,
    ).reshape(-1)

    cdf = torch.sigmoid(
        cutpoints.view(*([1] * latent_f.ndim), -1) - latent_f.unsqueeze(-1)
    )

    p_first = cdf[..., :1]
    p_middle = cdf[..., 1:] - cdf[..., :-1]
    p_last = 1.0 - cdf[..., -1:]

    probs = torch.cat([p_first, p_middle, p_last], dim=-1)

    probs = probs.clamp_min(eps)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(eps)

    return probs


def ordinal_expected_utility_from_latent(
    latent_f: Tensor,
    ordinal_likelihood,
    utility_values: Sequence[float] | Tensor,
    eps: float = 1e-12,
    cutpoints_getter: Optional[Callable[[object], Tensor]] = None,
) -> Tensor:
    """
    latent f samples から expected utility samples を計算する。

    Returns:
        expected utility with shape (..., q_like)
    """
    utilities = _canonicalize_utility_values(
        utility_values,
        device=latent_f.device,
        dtype=latent_f.dtype,
    )

    getter = cutpoints_getter or _get_cutpoints_from_ordinal_likelihood
    cutpoints = getter(ordinal_likelihood)

    probs = ordinal_logit_probs_from_latent(
        latent_f=latent_f,
        cutpoints=cutpoints,
        eps=eps,
    )

    if probs.shape[-1] != utilities.numel():
        raise RuntimeError(
            "Number of ordinal classes does not match utility_values length. "
            f"probs.shape[-1]={probs.shape[-1]}, "
            f"len(utility_values)={utilities.numel()}."
        )

    return (probs * utilities).sum(dim=-1)


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
) -> Tensor:
    if risk_type is None:
        return values_w.mean(dim=risk_dim)

    sorted_values = torch.sort(values_w, dim=risk_dim, descending=False).values

    k = max(1, int(math.ceil(int(n_w) * float(alpha))))
    tail = sorted_values.narrow(dim=risk_dim, start=0, length=k)

    if risk_type == "var":
        return tail.select(dim=risk_dim, index=k - 1)

    if risk_type == "cvar":
        return tail.mean(dim=risk_dim)

    raise ValueError(f"Unknown risk_type: {risk_type!r}.")


# ============================================================
# 1. Single-output ordinal expected utility objective
#    posterior latent samples -> expected utility
# ============================================================
class OrdinalExpectedUtilityMCObjective(MCAcquisitionObjective):
    """
    ordinal posterior samples を expected utility samples に変換する BO 用 objective。

    samples:
        sample_shape x batch_shape x q x 1
        または sample_shape x batch_shape x q
    return:
        sample_shape x batch_shape x q
    """

    def __init__(self, ordinal_likelihood, utility_values: Tensor) -> None:
        super().__init__()
        self.ordinal_likelihood = ordinal_likelihood
        self.register_buffer("utility_values", utility_values.reshape(-1))

    def forward(self, samples: Tensor, X: Tensor | None = None) -> Tensor:
        f = samples.squeeze(-1) if samples.shape[-1] == 1 else samples

        probs = self.ordinal_likelihood.class_probs_from_f(f)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        utilities = self.utility_values.to(device=probs.device, dtype=probs.dtype)
        return (probs * utilities).sum(dim=-1)

class OrdinalInputPerturbationExpectedUtilityObjective(MCAcquisitionObjective):
    """ordinal 用 objective。latent f を expected utility または score に変換・集約します。
    
    Args:
        ordinal_likelihood: ordinal latent f を class probability に変換する likelihood。
        utility_values: ordinal class ごとの utility 値。例: `[0.0, 1.0, 2.0]`。
        n_w: InputPerturbation で 1 点あたりに展開される摂動数。
        risk_type: InputPerturbation 集約の risk 種類。`None`、`var`、`cvar`。
        alpha: risk 集約または qNEHVI の近似設定に使うパラメータ。
        maximize: score が大きいほど良い向きに揃っているかどうか。
        aggregate_mean_when_no_risk: この acquisition / objective の動作を制御するパラメータ。
        allow_unexpanded: この acquisition / objective の動作を制御するパラメータ。
        eps: 数値安定化用の微小値。
        cutpoints_getter: ordinal likelihood から cutpoints を取得するための callable。
    
    Returns:
        Tensor: 入力 samples または score を変換・集約した objective value。
    
    Notes:
        ordinal class の意味は `utility_values` によって定義します。
    """

    def __init__(
        self,
        ordinal_likelihood,
        utility_values: Sequence[float] | Tensor,
        n_w: Optional[int] = None,
        risk_type: RiskType = None,
        alpha: float = 0.5,
        maximize: bool = True,
        aggregate_mean_when_no_risk: bool = True,
        allow_unexpanded: bool = True,
        eps: float = 1e-12,
        cutpoints_getter: Optional[Callable[[object], Tensor]] = None,
    ) -> None:
        super().__init__()

        self.ordinal_likelihood = ordinal_likelihood
        self.n_w = None if n_w is None else int(n_w)
        self.risk_type = risk_type
        self.alpha = float(alpha)
        self.maximize = bool(maximize)
        self.aggregate_mean_when_no_risk = bool(aggregate_mean_when_no_risk)
        self.allow_unexpanded = bool(allow_unexpanded)
        self.eps = float(eps)
        self.cutpoints_getter = cutpoints_getter

        utility_tensor = torch.as_tensor(utility_values, dtype=torch.double)

        if utility_tensor.ndim != 1:
            raise ValueError(
                "utility_values must be 1D. "
                f"Got shape={tuple(utility_tensor.shape)}."
            )

        self.register_buffer("utility_values", utility_tensor)

        _validate_n_w_risk(
            n_w=self.n_w,
            risk_type=self.risk_type,
            alpha=self.alpha,
        )

    def _latent_to_utility(self, samples: Tensor) -> Tensor:
        utilities = self.utility_values.to(
            device=samples.device,
            dtype=samples.dtype,
        )

        values = ordinal_expected_utility_from_latent(
            latent_f=samples,
            ordinal_likelihood=self.ordinal_likelihood,
            utility_values=utilities,
            eps=self.eps,
            cutpoints_getter=self.cutpoints_getter,
        )

        if not self.maximize:
            values = -values

        return values

    def forward(self, samples: Tensor, X: Optional[Tensor] = None) -> Tensor:
        if not torch.is_tensor(samples):
            raise TypeError(f"samples must be a Tensor. Got {type(samples)}.")

        values = self._latent_to_utility(samples)

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
                    maximize=True,
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
                    maximize=True,
                )

            if self.allow_unexpanded:
                return values

            raise RuntimeError(
                "Could not aggregate ordinal baseline samples. "
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
                    maximize=True,
                )

            if self.allow_unexpanded:
                return values

            raise RuntimeError(
                "Could not aggregate ordinal candidate samples. "
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
            maximize=True,
        )


# ============================================================
# 2. Multi-output ordinal expected utility objective
#    posterior latent samples -> expected utility
# ============================================================
def _extract_ordinal_likelihoods_from_model(
    model,
    ordinal_likelihoods=None,
):
    """
    multi-output ordinal model から ordinal likelihoods を解決する。

    優先順:
        1. 明示指定された ordinal_likelihoods
        2. model.ordinal_likelihoods
        3. model.likelihoods
        4. model.models[i].ordinal_likelihood
        5. model.models[i].likelihood
        6. model.ordinal_likelihood
        7. model.likelihood

    Args:
        model:
            MultiOutputOrdinalModel / ModelList など。
        ordinal_likelihoods:
            明示指定された likelihood または likelihood のリスト。

    Returns:
        list:
            ordinal likelihood のリスト。
    """
    if ordinal_likelihoods is not None:
        if isinstance(ordinal_likelihoods, (list, tuple)):
            return list(ordinal_likelihoods)
        return [ordinal_likelihoods]

    if model is None:
        raise ValueError(
            "Either model or ordinal_likelihoods must be provided."
        )

    if hasattr(model, "ordinal_likelihoods"):
        likelihoods = getattr(model, "ordinal_likelihoods")
        if callable(likelihoods):
            likelihoods = likelihoods()
        return list(likelihoods)

    if hasattr(model, "likelihoods"):
        likelihoods = getattr(model, "likelihoods")
        if callable(likelihoods):
            likelihoods = likelihoods()
        return list(likelihoods)

    if hasattr(model, "models"):
        likelihoods = []
        for i, submodel in enumerate(model.models):
            lik = getattr(submodel, "ordinal_likelihood", None)
            if lik is None:
                lik = getattr(submodel, "likelihood", None)

            if lik is None:
                raise ValueError(
                    f"Could not infer ordinal likelihood for submodel index {i}. "
                    "Each submodel should expose `ordinal_likelihood` or `likelihood`, "
                    "or pass ordinal_likelihoods explicitly."
                )

            likelihoods.append(lik)

        if len(likelihoods) == 0:
            raise ValueError("model.models is empty.")

        return likelihoods

    lik = getattr(model, "ordinal_likelihood", None)
    if lik is None:
        lik = getattr(model, "likelihood", None)

    if lik is not None:
        return [lik]

    raise ValueError(
        "Could not infer ordinal likelihoods from model. "
        "Expected one of: model.ordinal_likelihoods, model.likelihoods, "
        "model.models[i].ordinal_likelihood, model.models[i].likelihood, "
        "model.ordinal_likelihood, or model.likelihood."
    )

class MultiOutputOrdinalInputPerturbationObjective(MCMultiOutputObjective):
    """multi-output ordinal 用 objective。latent f を expected utility または score に変換・集約します。
    
    Args:
        ordinal_likelihoods: multi-output ordinal の各出力に対応する likelihood のリスト。
        utility_values: ordinal class ごとの utility 値。例: `[0.0, 1.0, 2.0]`。
        n_w: InputPerturbation で 1 点あたりに展開される摂動数。
        risk_type: InputPerturbation 集約の risk 種類。`None`、`var`、`cvar`。
        alpha: risk 集約または qNEHVI の近似設定に使うパラメータ。
        maximize: score が大きいほど良い向きに揃っているかどうか。
        aggregate_mean_when_no_risk: この acquisition / objective の動作を制御するパラメータ。
        allow_unexpanded: この acquisition / objective の動作を制御するパラメータ。
        eps: 数値安定化用の微小値。
        cutpoints_getter: ordinal likelihood から cutpoints を取得するための callable。
    
    Returns:
        Tensor: 入力 samples または score を変換・集約した objective value。
    
    Notes:
        ordinal class の意味は `utility_values` によって定義します。
    """

    def __init__(
        self,
        utility_values: Sequence[float] | Tensor | Sequence[Sequence[float]],
        ordinal_likelihoods=None,
        model=None,
        n_w: Optional[int] = None,
        risk_type: RiskType = None,
        alpha: float = 0.5,
        maximize: bool | Sequence[bool] = True,
        aggregate_mean_when_no_risk: bool = True,
        allow_unexpanded: bool = True,
        eps: float = 1e-12,
        cutpoints_getter: Optional[Callable[[object], Tensor]] = None,
    ) -> None:
        super().__init__()
    
        self.ordinal_likelihoods = _extract_ordinal_likelihoods_from_model(
            model=model,
            ordinal_likelihoods=ordinal_likelihoods,
        )
    
        self.n_w = None if n_w is None else int(n_w)
        self.risk_type = risk_type
        self.alpha = float(alpha)
        self.aggregate_mean_when_no_risk = bool(aggregate_mean_when_no_risk)
        self.allow_unexpanded = bool(allow_unexpanded)
        self.eps = float(eps)
        self.cutpoints_getter = cutpoints_getter
    
        utility_tensor = torch.as_tensor(utility_values, dtype=torch.double)
    
        if utility_tensor.ndim not in (1, 2):
            raise ValueError(
                "utility_values must be 1D [K] or 2D [m, K]. "
                f"Got shape={tuple(utility_tensor.shape)}."
            )
    
        self.register_buffer("utility_values", utility_tensor)
    
        if isinstance(maximize, bool):
            maximize_tensor = torch.tensor([maximize], dtype=torch.bool)
        else:
            maximize_tensor = torch.as_tensor(list(maximize), dtype=torch.bool)
    
            if maximize_tensor.ndim != 1:
                raise ValueError("maximize must be bool or 1D sequence of bool.")
    
        self.register_buffer("maximize_tensor", maximize_tensor)
    
        _validate_n_w_risk(
            n_w=self.n_w,
            risk_type=self.risk_type,
            alpha=self.alpha,
        )

    def _get_likelihood_for_output(self, j: int):
        if len(self.ordinal_likelihoods) == 1:
            return self.ordinal_likelihoods[0]

        return self.ordinal_likelihoods[j]

    def _get_utilities_for_output(
        self,
        j: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        utilities = self.utility_values.to(device=device, dtype=dtype)

        if utilities.ndim == 1:
            return utilities

        return utilities[j]

    def _get_maximize_for_output(
        self,
        j: int,
        *,
        device: torch.device,
    ) -> bool:
        maximize_tensor = self.maximize_tensor.to(device=device)

        if maximize_tensor.numel() == 1:
            return bool(maximize_tensor.item())

        return bool(maximize_tensor[j].item())

    def _latent_to_utility(self, samples: Tensor) -> Tensor:
        if samples.ndim < 2:
            raise RuntimeError(
                "multi-output ordinal samples must have shape (..., q_like, m). "
                f"Got shape={tuple(samples.shape)}."
            )

        m = samples.shape[-1]

        if len(self.ordinal_likelihoods) not in (1, m):
            raise RuntimeError(
                "Number of ordinal_likelihoods must be 1 or match output dimension m. "
                f"len(ordinal_likelihoods)={len(self.ordinal_likelihoods)}, m={m}."
            )

        if self.utility_values.ndim == 2 and self.utility_values.shape[0] != m:
            raise RuntimeError(
                "If utility_values is 2D, its first dimension must match output dimension m. "
                f"utility_values.shape={tuple(self.utility_values.shape)}, m={m}."
            )

        if self.maximize_tensor.numel() not in (1, m):
            raise RuntimeError(
                "maximize must be bool or have length m. "
                f"len(maximize)={self.maximize_tensor.numel()}, m={m}."
            )

        values = []

        for j in range(m):
            latent_j = samples[..., j]

            utilities_j = self._get_utilities_for_output(
                j,
                device=samples.device,
                dtype=samples.dtype,
            )

            likelihood_j = self._get_likelihood_for_output(j)

            value_j = ordinal_expected_utility_from_latent(
                latent_f=latent_j,
                ordinal_likelihood=likelihood_j,
                utility_values=utilities_j,
                eps=self.eps,
                cutpoints_getter=self.cutpoints_getter,
            )

            if not self._get_maximize_for_output(j, device=samples.device):
                value_j = -value_j

            values.append(value_j)

        return torch.stack(values, dim=-1)

    def forward(self, samples: Tensor, X: Optional[Tensor] = None) -> Tensor:
        if not torch.is_tensor(samples):
            raise TypeError(f"samples must be a Tensor. Got {type(samples)}.")

        values = self._latent_to_utility(samples)

        if values.ndim < 2:
            raise RuntimeError(
                "values must have at least shape (..., q_like, m). "
                f"Got shape={tuple(values.shape)}."
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
                return _aggregate_multioutput_axis(
                    values,
                    n_w=n_w,
                    risk_type=self.risk_type,
                    alpha=self.alpha,
                    risk_dim=-2,
                )

            q_like = values.shape[-2]

            if q_like == n * n_w:
                values_w = values.reshape(*values.shape[:-2], n, n_w, m)
                return _aggregate_multioutput_axis(
                    values_w,
                    n_w=n_w,
                    risk_type=self.risk_type,
                    alpha=self.alpha,
                    risk_dim=-2,
                )

            if self.allow_unexpanded:
                return values

            raise RuntimeError(
                "Could not aggregate multi-output ordinal baseline samples. "
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
                return _aggregate_multioutput_axis(
                    values_w,
                    n_w=n_w,
                    risk_type=self.risk_type,
                    alpha=self.alpha,
                    risk_dim=-2,
                )

            if self.allow_unexpanded:
                return values

            raise RuntimeError(
                "Could not aggregate multi-output ordinal candidate samples. "
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

        return _aggregate_multioutput_axis(
            values_w,
            n_w=n_w,
            risk_type=self.risk_type,
            alpha=self.alpha,
            risk_dim=-2,
        )


# ============================================================
# 3. Single-output ordinal score objective
# ============================================================


class OrdinalScoreObjective(nn.Module):
    """ordinal 用 objective。latent f を expected utility または score に変換・集約します。
    
    Args:
        n_w: InputPerturbation で 1 点あたりに展開される摂動数。
        risk_type: InputPerturbation 集約の risk 種類。`None`、`var`、`cvar`。
        alpha: risk 集約または qNEHVI の近似設定に使うパラメータ。
        maximize: score が大きいほど良い向きに揃っているかどうか。
        weight: score または objective に掛ける重み。
        sign: 目的の向きを揃える符号。最大化なら +1、最小化なら -1 を使います。
        aggregated_risk_mode: この acquisition / objective の動作を制御するパラメータ。
    
    Returns:
        Tensor: 入力 samples または score を変換・集約した objective value。
    
    Notes:
        ordinal class の意味は `utility_values` によって定義します。
    """

    def __init__(
        self,
        n_w: Optional[int] = None,
        risk_type: RiskType = None,
        alpha: float = 0.5,
        maximize: bool = True,
        weight: float = 1.0,
        sign: float = 1.0,
        aggregated_risk_mode: AggregatedRiskMode = "ignore",
    ) -> None:
        super().__init__()

        self.n_w = None if n_w is None else int(n_w)
        self.risk_type = risk_type
        self.alpha = float(alpha)
        self.maximize = bool(maximize)
        self.weight = float(weight)
        self.sign = float(sign)
        self.aggregated_risk_mode = aggregated_risk_mode

        _validate_n_w_risk(
            n_w=self.n_w,
            risk_type=self.risk_type,
            alpha=self.alpha,
        )

        if self.aggregated_risk_mode not in ("ignore", "error"):
            raise ValueError("aggregated_risk_mode must be 'ignore' or 'error'.")

    @staticmethod
    def _ensure_q_batch(X: Tensor) -> Tensor:
        return X if X.dim() > 2 else X.unsqueeze(0)

    def _is_aggregated_score(self, score: Tensor, X: Optional[Tensor]) -> bool:
        if X is None or score.ndim == 0:
            return False

        Xq = self._ensure_q_batch(X)
        return tuple(score.shape) == tuple(Xq.shape[:-2])

    def _is_pointwise_score(self, score: Tensor, X: Optional[Tensor]) -> bool:
        if X is None or score.ndim == 0:
            return True

        Xq = self._ensure_q_batch(X)
        return tuple(score.shape[:-1]) == tuple(Xq.shape[:-2])

    def forward(self, score: Tensor, X: Optional[Tensor] = None) -> Tensor:
        if not torch.is_tensor(score):
            raise TypeError(f"score must be a Tensor. Got {type(score)}.")

        score = score * self.sign * self.weight

        if score.ndim == 0:
            return score

        if self.n_w is None or self.n_w <= 1:
            return score

        if self._is_aggregated_score(score, X):
            if self.aggregated_risk_mode == "error":
                raise RuntimeError(
                    "OrdinalScoreObjective received an aggregated / joint score "
                    f"with shape={tuple(score.shape)}. n_w aggregation is only valid "
                    "for pointwise score with shape (*batch, q * n_w)."
                )

            return score

        if not self._is_pointwise_score(score, X):
            raise RuntimeError(
                "OrdinalScoreObjective expected either an aggregated score "
                "or a pointwise score with shape (*batch, q_like). "
                f"Got score.shape={tuple(score.shape)}, "
                f"X.shape={None if X is None else tuple(X.shape)}."
            )

        q_expanded = score.shape[-1]

        if q_expanded % int(self.n_w) != 0:
            raise RuntimeError(
                "score.shape[-1] must be divisible by n_w. "
                f"Got score.shape={tuple(score.shape)}, n_w={self.n_w}."
            )

        q = q_expanded // int(self.n_w)
        score_w = score.reshape(*score.shape[:-1], q, int(self.n_w))

        return _aggregate_scalar_axis(
            score_w,
            n_w=int(self.n_w),
            risk_type=self.risk_type,
            alpha=self.alpha,
            risk_dim=-1,
            maximize=self.maximize,
        )


# ============================================================
# 4. Multi-output ordinal score objective
# ============================================================


class MultiOutputOrdinalScoreObjective(nn.Module):
    """multi-output ordinal 用 objective。latent f を expected utility または score に変換・集約します。
    
    Args:
        n_w: InputPerturbation で 1 点あたりに展開される摂動数。
        risk_type: InputPerturbation 集約の risk 種類。`None`、`var`、`cvar`。
        alpha: risk 集約または qNEHVI の近似設定に使うパラメータ。
        aggregated_risk_mode: この acquisition / objective の動作を制御するパラメータ。
        score_shape_mode: この acquisition / objective の動作を制御するパラメータ。
    
    Returns:
        Tensor: 入力 samples または score を変換・集約した objective value。
    
    Notes:
        ordinal class の意味は `utility_values` によって定義します。
    """

    def __init__(
        self,
        n_w: Optional[int] = None,
        risk_type: RiskType = None,
        alpha: float = 0.5,
        aggregated_risk_mode: AggregatedRiskMode = "ignore",
        score_shape_mode: OrdinalScoreShapeMode = "auto",
    ) -> None:
        super().__init__()

        self.n_w = None if n_w is None else int(n_w)
        self.risk_type = risk_type
        self.alpha = float(alpha)
        self.aggregated_risk_mode = aggregated_risk_mode
        self.score_shape_mode = score_shape_mode

        _validate_n_w_risk(
            n_w=self.n_w,
            risk_type=self.risk_type,
            alpha=self.alpha,
        )

        if self.aggregated_risk_mode not in ("ignore", "error"):
            raise ValueError("aggregated_risk_mode must be 'ignore' or 'error'.")

        if self.score_shape_mode not in (
            "auto",
            "pointwise",
            "multioutput_qm",
            "multioutput_mq",
            "aggregated",
        ):
            raise ValueError(
                "score_shape_mode must be one of "
                "'auto', 'pointwise', 'multioutput_qm', "
                "'multioutput_mq', or 'aggregated'."
            )

    @staticmethod
    def _ensure_q_batch(X: Tensor) -> Tensor:
        return X if X.dim() > 2 else X.unsqueeze(0)

    def _batch_shape_from_X(self, X: Optional[Tensor]) -> Optional[torch.Size]:
        if X is None:
            return None

        Xq = self._ensure_q_batch(X)
        return Xq.shape[:-2]

    def _q_from_X(self, X: Optional[Tensor]) -> Optional[int]:
        if X is None:
            return None

        Xq = self._ensure_q_batch(X)
        return int(Xq.shape[-2])

    def _infer_score_shape_mode(
        self,
        score: Tensor,
        X: Optional[Tensor],
    ) -> OrdinalScoreShapeMode:
        if self.score_shape_mode != "auto":
            return self.score_shape_mode

        if score.ndim == 0:
            return "aggregated"

        batch_shape = self._batch_shape_from_X(X)
        q = self._q_from_X(X)

        if batch_shape is not None:
            if tuple(score.shape) == tuple(batch_shape):
                return "aggregated"

            if score.ndim >= 1 and tuple(score.shape[:-1]) == tuple(batch_shape):
                return "pointwise"

            if score.ndim >= 2 and tuple(score.shape[:-2]) == tuple(batch_shape):
                if q is not None and self.n_w is not None:
                    q_expanded = q * int(self.n_w)

                    if score.shape[-2] in (q, q_expanded):
                        return "multioutput_qm"

                    if score.shape[-1] in (q, q_expanded):
                        return "multioutput_mq"

                return "multioutput_qm"

        if score.ndim == 1:
            return "pointwise"

        if score.ndim >= 2:
            return "multioutput_qm"

        return "aggregated"

    def _handle_aggregated_score(self, score: Tensor) -> Tensor:
        if (
            self.n_w is not None
            and self.n_w > 1
            and self.aggregated_risk_mode == "error"
        ):
            raise RuntimeError(
                "MultiOutputOrdinalScoreObjective received an aggregated / joint score "
                f"with shape={tuple(score.shape)}. InputPerturbation aggregation is only valid "
                "for pointwise scores with shape (*batch, q * n_w), "
                "(*batch, q * n_w, m), or (*batch, m, q * n_w)."
            )

        return score

    def _aggregate_pointwise_score(self, score: Tensor) -> Tensor:
        if self.n_w is None or self.n_w <= 1:
            return score

        q_expanded = score.shape[-1]

        if q_expanded % int(self.n_w) != 0:
            raise RuntimeError(
                "score.shape[-1] must be divisible by n_w. "
                f"Got score.shape={tuple(score.shape)}, n_w={self.n_w}."
            )

        q = q_expanded // int(self.n_w)
        score_w = score.reshape(*score.shape[:-1], q, int(self.n_w))

        return _aggregate_scalar_axis(
            score_w,
            n_w=int(self.n_w),
            risk_type=self.risk_type,
            alpha=self.alpha,
            risk_dim=-1,
            maximize=True,
        )

    def _aggregate_multioutput_qm_score(self, score: Tensor) -> Tensor:
        if self.n_w is None or self.n_w <= 1:
            return score

        q_expanded = score.shape[-2]
        m = score.shape[-1]

        if q_expanded % int(self.n_w) != 0:
            raise RuntimeError(
                "score.shape[-2] must be divisible by n_w. "
                f"Got score.shape={tuple(score.shape)}, n_w={self.n_w}."
            )

        q = q_expanded // int(self.n_w)
        score_w = score.reshape(*score.shape[:-2], q, int(self.n_w), m)

        return _aggregate_multioutput_axis(
            score_w,
            n_w=int(self.n_w),
            risk_type=self.risk_type,
            alpha=self.alpha,
            risk_dim=-2,
        )

    def _aggregate_multioutput_mq_score(self, score: Tensor) -> Tensor:
        if score.ndim < 2:
            raise RuntimeError(
                "multioutput_mq score must have shape (*batch, m, q_like). "
                f"Got shape={tuple(score.shape)}."
            )

        if self.n_w is None or self.n_w <= 1:
            return score.transpose(-1, -2)

        m = score.shape[-2]
        q_expanded = score.shape[-1]

        if q_expanded % int(self.n_w) != 0:
            raise RuntimeError(
                "score.shape[-1] must be divisible by n_w. "
                f"Got score.shape={tuple(score.shape)}, n_w={self.n_w}."
            )

        q = q_expanded // int(self.n_w)

        score_w = score.reshape(*score.shape[:-2], m, q, int(self.n_w))
        score_mq = _aggregate_scalar_axis(
            score_w,
            n_w=int(self.n_w),
            risk_type=self.risk_type,
            alpha=self.alpha,
            risk_dim=-1,
            maximize=True,
        )

        return score_mq.transpose(-1, -2)

    def forward(self, score: Tensor, X: Optional[Tensor] = None) -> Tensor:
        if not torch.is_tensor(score):
            raise TypeError(f"score must be a Tensor. Got {type(score)}.")

        mode = self._infer_score_shape_mode(score, X)

        if mode == "aggregated":
            return self._handle_aggregated_score(score)

        if mode == "pointwise":
            return self._aggregate_pointwise_score(score)

        if mode == "multioutput_qm":
            return self._aggregate_multioutput_qm_score(score)

        if mode == "multioutput_mq":
            return self._aggregate_multioutput_mq_score(score)

        raise RuntimeError(f"Unsupported inferred score shape mode: {mode}")


# ============================================================
# 5. Objective mixins
# ============================================================


class OrdinalScoreObjectiveMixin:
    """ordinal 用 objective。latent f を expected utility または score に変換・集約します。
    
    Returns:
        Tensor: 入力 samples または score を変換・集約した objective value。
    
    Notes:
        ordinal class の意味は `utility_values` によって定義します。
    """

    objective: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]]

    def _set_ordinal_score_objective(
        self,
        objective: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
    ) -> None:
        self.objective = objective

    def _apply_objective_to_score(
        self,
        score: Tensor,
        X: Optional[Tensor],
        name: str,
    ) -> Tensor:
        if self.objective is None:
            return score

        out = self.objective(score, X=X)

        if not torch.is_tensor(out):
            raise TypeError(f"{name}: objective must return a Tensor. Got {type(out)}.")

        return out


class MultiOutputOrdinalScoreObjectiveMixin:
    """multi-output ordinal 用 objective。latent f を expected utility または score に変換・集約します。
    
    Returns:
        Tensor: 入力 samples または score を変換・集約した objective value。
    
    Notes:
        ordinal class の意味は `utility_values` によって定義します。
    """

    objective: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]]

    def _set_multioutput_ordinal_score_objective(
        self,
        objective: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
    ) -> None:
        self.objective = objective

    def _apply_objective_to_multioutput_score(
        self,
        score: Tensor,
        X: Optional[Tensor],
        name: str,
    ) -> Tensor:
        if self.objective is None:
            return score

        out = self.objective(score, X=X)

        if not torch.is_tensor(out):
            raise TypeError(f"{name}: objective must return a Tensor. Got {type(out)}.")

        return out

__all__ = [
    "ordinal_logit_probs_from_latent",
    "ordinal_expected_utility_from_latent",
    "OrdinalInputPerturbationExpectedUtilityObjective",
    "MultiOutputOrdinalInputPerturbationObjective",
    "OrdinalScoreObjective",
    "MultiOutputOrdinalScoreObjective",
    "OrdinalScoreObjectiveMixin",
    "MultiOutputOrdinalScoreObjectiveMixin",
]
