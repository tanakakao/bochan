from __future__ import annotations

import math
from typing import Callable, Literal, Optional, Sequence

import torch
from torch import Tensor

from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.models.model import Model
from botorch.utils.transforms import t_batch_mode_transform

from ..hetero_utils import (
    _expand_scalar_or_list,
    _normal_cdf,
    aggregate_objectives,
    make_weight_tensor,
    stack_multi_summaries,
)


RiskType = Optional[Literal["var", "cvar"]]
ReductionType = Literal["mean", "sum", "max"]
OutputMode = Literal["mean", "sum", "max", "min", "weighted_mean"]
BoundaryReduction = Literal["sum", "mean", "max", "min"]
NoiseWeightMode = Literal["none", "inverse_linear", "inverse_sqrt", "inverse_exp", "custom"]
NoiseCombineType = Literal["multiply", "subtract"]
ROIWeightMode = Literal[
    "none",
    "mean_above",
    "mean_below",
    "mean_interval",
    "poe_above",
    "poe_below",
    "poe_interval",
    "custom",
]
ROICombineType = Literal["multiply", "add"]


# =========================================================
# Score objective
# =========================================================
class HeteroMultiOutputOrdinalLevelSetScoreObjective(torch.nn.Module):
    """
    multi-output hetero ordinal level-set acquisition の pointwise score に作用する objective。

    posterior samples ではなく、Probability of Exceedance / Level-set uncertainty /
    Straddle / BoundaryVariance などで計算済みの score に作用する。
    主な用途は InputPerturbation の q * n_w を q に戻すこと。
    """

    def __init__(
        self,
        n_w: Optional[int] = None,
        risk_type: RiskType = None,
        alpha: float = 0.5,
        maximize: bool = True,
        weight: float = 1.0,
        sign: float = 1.0,
        aggregated_risk_mode: Literal["ignore", "error"] = "ignore",
    ) -> None:
        super().__init__()
        self.n_w = None if n_w is None else int(n_w)
        self.risk_type = risk_type
        self.alpha = float(alpha)
        self.maximize = bool(maximize)
        self.weight = float(weight)
        self.sign = float(sign)
        self.aggregated_risk_mode = aggregated_risk_mode

        if self.n_w is not None and self.n_w <= 0:
            raise ValueError("n_w must be positive or None.")
        if self.risk_type not in (None, "var", "cvar"):
            raise ValueError(f"Unknown risk_type: {self.risk_type!r}.")
        if self.risk_type is not None and self.n_w is None:
            raise ValueError("risk_type is specified, but n_w is None.")
        if self.risk_type is not None and not (0.0 < self.alpha <= 1.0):
            raise ValueError("alpha must be in (0, 1].")
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
                    "HeteroMultiOutputOrdinalLevelSetScoreObjective received an aggregated score. "
                    "InputPerturbation aggregation requires pointwise score."
                )
            return score

        q_expanded = int(score.shape[-1])
        if q_expanded % int(self.n_w) != 0:
            raise RuntimeError(
                "score.shape[-1] must be divisible by n_w. "
                f"Got score.shape={tuple(score.shape)}, n_w={self.n_w}."
            )

        q = q_expanded // int(self.n_w)
        score_w = score.reshape(*score.shape[:-1], q, int(self.n_w))

        if self.risk_type is None:
            return score_w.mean(dim=-1)

        # acquisition score は大きいほどよい。maximize=True では worst tail は小さい側。
        descending = not self.maximize
        sorted_score = torch.sort(score_w, dim=-1, descending=descending).values
        k = max(1, int(math.ceil(int(self.n_w) * self.alpha)))
        tail = sorted_score[..., :k]

        if self.risk_type == "var":
            return tail[..., -1]
        if self.risk_type == "cvar":
            return tail.mean(dim=-1)
        raise ValueError(f"Unknown risk_type: {self.risk_type!r}.")


# Backward-compatible internal name; not a public acquisition alias.
_MultiObjectiveHeteroOrdinalLevelSetScoreObjective = (
    HeteroMultiOutputOrdinalLevelSetScoreObjective
)


# =========================================================
# Generic tensor helpers
# =========================================================
def _ensure_q_batch(X: Tensor) -> Tensor:
    if not torch.is_tensor(X):
        raise TypeError(f"X must be Tensor. Got {type(X)}.")
    if X.ndim == 1:
        return X.view(1, 1, -1)
    if X.ndim == 2:
        return X.unsqueeze(0)
    return X


def _reduce_q(score: Tensor, reduction: ReductionType) -> Tensor:
    if score.ndim == 0:
        return score
    if score.shape[-1] == 1:
        return score.squeeze(-1)
    if reduction == "mean":
        return score.mean(dim=-1)
    if reduction == "sum":
        return score.sum(dim=-1)
    if reduction == "max":
        return score.max(dim=-1).values
    raise ValueError(f"Unknown reduction: {reduction!r}.")


def _coerce_reference_to_tensor(X_ref, *, ref: Optional[Tensor] = None) -> Optional[Tensor]:
    if X_ref is None:
        return None

    if torch.is_tensor(X_ref):
        out = X_ref
    elif isinstance(X_ref, (list, tuple)):
        tensors = []
        for item in X_ref:
            if item is None:
                continue
            t = _coerce_reference_to_tensor(item, ref=ref)
            if t is not None and t.numel() > 0:
                tensors.append(t)
        if len(tensors) == 0:
            return None
        if len(tensors) == 1:
            out = tensors[0]
        else:
            try:
                out = torch.cat(tensors, dim=-2)
            except RuntimeError:
                out = torch.cat([t.reshape(-1, t.shape[-1]) for t in tensors], dim=-2)
    else:
        raise TypeError(
            "X_pending must be None, Tensor, list, or tuple. "
            f"Got {type(X_ref)}."
        )

    if ref is not None:
        out = out.to(device=ref.device, dtype=ref.dtype)
    return out


def _align_pointwise_score_to_X(
    score: Tensor,
    Xt: Tensor,
    *,
    name: str,
    reduce_extra: Literal["mean", "sum"] = "mean",
) -> Tensor:
    """
    score を Xt.shape[:-1] = (*batch, q_like) に揃える。

    SAAS / fully Bayesian / MC extra dim などが前に乗る場合は平均または和で潰す。
    """
    target = Xt.shape[:-1]
    out = score

    if out.shape == target:
        return out

    if out.ndim >= 1 and out.shape[-1] == 1 and len(target) >= 1 and target[-1] != 1:
        out = out.squeeze(-1)
        if out.shape == target:
            return out

    if out.numel() == int(torch.tensor(target).prod().item()):
        return out.reshape(target)

    while out.ndim > len(target):
        out = out.mean(dim=0) if reduce_extra == "mean" else out.sum(dim=0)
        if out.shape == target:
            return out

    if out.shape == target:
        return out

    if out.ndim == len(target) and out.shape[-1] == target[-1]:
        try:
            return out.expand(target)
        except RuntimeError:
            pass

    raise RuntimeError(
        f"{name}: score shape mismatch. "
        f"score.shape={tuple(score.shape)}, expected={tuple(target)}, Xt.shape={tuple(Xt.shape)}."
    )


def _objective_call(objective, score: Tensor, X: Optional[Tensor]):
    try:
        return objective(score, X=X)
    except TypeError:
        return objective(score)


def _apply_multioutput_hetero_ordinal_levelset_objective_to_score(
    owner,
    score: Tensor,
    X: Optional[Tensor] = None,
    name: str = "HeteroMultiOutputOrdinalLevelSet",
) -> Tensor:
    objective = getattr(owner, "objective", None)
    if objective is None:
        return score

    out = _objective_call(objective, score, X)
    if not torch.is_tensor(out):
        raise RuntimeError(f"{name}: objective must return a Tensor. Got {type(out)}.")
    return out



# =========================================================
# Ordinal boundary helpers
# =========================================================
def _try_call_zero_arg(obj):
    return obj() if callable(obj) else obj


def _is_ordinal_likelihood(obj) -> bool:
    return obj is not None and (
        hasattr(obj, "marginal_class_probs")
        or hasattr(obj, "class_probs_from_f")
        or hasattr(obj, "transformed_cutpoints")
        or hasattr(obj, "cutpoints")
        or hasattr(obj, "raw_cutpoints")
    )


def _get_ordinal_likelihood(model: Model):
    for cand in (
        getattr(model, "ordinal_likelihood", None),
        getattr(model, "likelihood", None),
    ):
        if _is_ordinal_likelihood(cand):
            return cand
    raise ValueError(
        "Each submodel must expose ordinal_likelihood or likelihood with cutpoints."
    )


def _get_cutpoints_from_likelihood(ordinal_likelihood) -> Tensor:
    """ordinal likelihood から cutpoints を取得し、定数として detach して返す。"""
    if hasattr(ordinal_likelihood, "get_cutpoints"):
        cutpoints = _try_call_zero_arg(getattr(ordinal_likelihood, "get_cutpoints"))
        return torch.as_tensor(cutpoints).detach().clone().reshape(-1)

    for name in ("transformed_cutpoints", "cutpoints", "thresholds", "cuts", "cutoffs"):
        if hasattr(ordinal_likelihood, name):
            cutpoints = _try_call_zero_arg(getattr(ordinal_likelihood, name))
            return torch.as_tensor(cutpoints).detach().clone().reshape(-1)

    if hasattr(ordinal_likelihood, "raw_cutpoints"):
        raw = torch.as_tensor(
            _try_call_zero_arg(getattr(ordinal_likelihood, "raw_cutpoints"))
        ).detach().clone()
        if hasattr(ordinal_likelihood, "transform_cutpoints"):
            cutpoints = ordinal_likelihood.transform_cutpoints(raw)
            return torch.as_tensor(cutpoints).detach().clone().reshape(-1)
        return torch.sort(raw.reshape(-1)).values.detach().clone()

    raise ValueError(
        "Could not find cutpoints on ordinal likelihood. Expected get_cutpoints, "
        "transformed_cutpoints, cutpoints, thresholds, cuts, cutoffs, or raw_cutpoints."
    )


def _to_optional_list(value, n: int, *, name: str) -> list:
    if value is None:
        return [None] * n
    if isinstance(value, (list, tuple)):
        if len(value) != n:
            raise ValueError(
                f"{name} length must match number of outputs. Expected {n}, got {len(value)}."
            )
        return list(value)
    return [value] * n


def _prepare_boundary_weights(
    boundary_weights: Optional[Tensor | Sequence[float]],
    n_boundaries: int,
    *,
    device,
    dtype,
) -> Optional[Tensor]:
    if boundary_weights is None:
        return None
    w = torch.as_tensor(boundary_weights, device=device, dtype=dtype).detach().reshape(-1)
    if w.numel() != n_boundaries:
        raise ValueError(
            f"boundary_weights must have length {n_boundaries}, got {w.numel()}."
        )
    return w


def _aggregate_boundary_scores(
    boundary_scores: Tensor,
    *,
    target_boundary_idx: Optional[int] = None,
    boundary_weights: Optional[Tensor | Sequence[float]] = None,
    boundary_reduction: BoundaryReduction = "sum",
) -> Tensor:
    """
    boundary-wise score を pointwise score に集約する。

    ``target_boundary_idx=k`` は class k / class k+1 境界に対応する。
    例: class 0/1/2 では idx=0 が 0/1 境界、idx=1 が 1/2 境界。
    """
    if boundary_scores.ndim < 1:
        raise RuntimeError("boundary_scores must have a boundary dimension.")

    n_boundaries = boundary_scores.shape[-1]

    if target_boundary_idx is not None:
        idx = int(target_boundary_idx)
        if not (0 <= idx < n_boundaries):
            raise ValueError(
                f"target_boundary_idx must be in [0, {n_boundaries - 1}], got {idx}."
            )
        return boundary_scores[..., idx]

    w = _prepare_boundary_weights(
        boundary_weights,
        n_boundaries,
        device=boundary_scores.device,
        dtype=boundary_scores.dtype,
    )
    if w is not None:
        boundary_scores = boundary_scores * w.view(
            *([1] * (boundary_scores.ndim - 1)),
            -1,
        )

    if boundary_reduction == "sum":
        return boundary_scores.sum(dim=-1)
    if boundary_reduction == "mean":
        return boundary_scores.mean(dim=-1)
    if boundary_reduction == "max":
        return boundary_scores.max(dim=-1).values
    if boundary_reduction == "min":
        return boundary_scores.min(dim=-1).values

    raise ValueError(f"Unknown boundary_reduction: {boundary_reduction!r}.")


def _boundary_kernel_scores(values: Tensor, cutpoints: Tensor, tau: float | Tensor) -> Tensor:
    cp = cutpoints.detach().to(device=values.device, dtype=values.dtype).reshape(-1)
    tau_t = torch.as_tensor(tau, device=values.device, dtype=values.dtype).clamp_min(1e-8)
    z2 = ((values.unsqueeze(-1) - cp.view(*([1] * values.ndim), -1)) / tau_t) ** 2
    return torch.exp(-0.5 * z2)


# =========================================================
# Classification-aligned base
# =========================================================
class _BaseHeteroMultiOutputOrdinalLevelSetAcquisition(AcquisitionFunction):
    """
    heteroscedastic multi-output ordinal level-set base。

    Standard order:
        per-output pointwise score
        -> ROI weighting per-output
        -> noise weighting per-output
        -> output aggregation
        -> pending penalty
        -> objective
        -> q reduction

    以前の ordinal 実装の `@concatenate_pending_points` は使わず、
    classification multi-output level-set 側と同じ penalty 型に寄せる。
    """

    def __init__(
        self,
        model: Model,
        *,
        utility_values_list: Optional[Sequence[Optional[Sequence[float] | Tensor]]] = None,
        objective_weights: Optional[Sequence[float] | Tensor] = None,
        variance_scale: float | Sequence[float] | Tensor = 1.0,
        tau: float | Sequence[float] | Tensor = 1e-6,
        default_sigma: float | Sequence[float] | Tensor = 0.0,
        reduction: ReductionType = "mean",
        # backward-compatible aliases
        reduce: Optional[str] = None,
        output_mode: OutputMode = "mean",
        aggregate: Optional[str] = None,
        eps: float = 1e-12,
        # pending penalty
        pending_penalty_weight: float = 0.0,
        pending_penalty_beta: float = 10.0,
        X_pending: Optional[Tensor] = None,
        # ROI
        roi_mode: ROIWeightMode = "none",
        roi_combine: ROICombineType = "multiply",
        roi_threshold: float = 0.5,
        roi_interval: Optional[tuple[float, float]] = None,
        roi_beta: float = 20.0,
        roi_bandwidth: float = 0.15,
        roi_min_weight: float = 0.0,
        roi_weight_scale: float = 1.0,
        roi_weight_fn: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
        # noise
        noise_mode: NoiseWeightMode = "inverse_linear",
        noise_combine: NoiseCombineType = "multiply",
        noise_penalty_lambda: float = 1.0,
        noise_min_weight: float = 0.0,
        noise_weight_scale: float = 1.0,
        noise_event_aggregate: OutputMode = "mean",
        noise_weight_fn: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
        # old compatibility
        noise_penalty: float | Sequence[float] | Tensor = 0.0,
        objective: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
    ) -> None:
        super().__init__(model=model)

        if not hasattr(model, "models"):
            raise ValueError("Hetero multi-output ordinal level-set acquisition expects model.models.")

        self.submodels = list(model.models)
        self.m = len(self.submodels)
        if self.m == 0:
            raise ValueError("model.models is empty.")

        self.ordinal_likelihoods = [_get_ordinal_likelihood(sm) for sm in self.submodels]
        # cutpoints は likelihood parameter から計算されることがあり、
        # grad_fn を持つ Tensor をそのまま保持すると optimizer step 間で
        # graph 再利用エラーが起きるため、定数として detach して保持する。
        self.cutpoints_list = [
            _get_cutpoints_from_likelihood(lik).detach().clone()
            for lik in self.ordinal_likelihoods
        ]

        if reduce is not None:
            reduction = str(reduce)
        if aggregate is not None:
            output_mode = str(aggregate)

        if reduction not in ("mean", "sum", "max"):
            raise ValueError("reduction must be 'mean', 'sum', or 'max'.")
        if output_mode not in ("mean", "sum", "max", "min", "weighted_mean"):
            raise ValueError("output_mode must be one of mean/sum/max/min/weighted_mean.")
        if noise_event_aggregate not in ("mean", "sum", "max", "min", "weighted_mean"):
            raise ValueError("noise_event_aggregate must be one of mean/sum/max/min/weighted_mean.")
        if roi_mode not in (
            "none",
            "mean_above",
            "mean_below",
            "mean_interval",
            "poe_above",
            "poe_below",
            "poe_interval",
            "custom",
        ):
            raise ValueError(f"Unknown roi_mode: {roi_mode!r}.")
        if roi_combine not in ("multiply", "add"):
            raise ValueError("roi_combine must be 'multiply' or 'add'.")
        if noise_mode not in ("none", "inverse_linear", "inverse_sqrt", "inverse_exp", "custom"):
            raise ValueError(f"Unknown noise_mode: {noise_mode!r}.")
        if noise_combine not in ("multiply", "subtract"):
            raise ValueError("noise_combine must be 'multiply' or 'subtract'.")

        self.utility_values_list = utility_values_list
        self.objective_weights = objective_weights
        self.variance_scale = variance_scale
        self.tau = tau
        self.default_sigma = default_sigma
        self.reduction = reduction
        self.output_mode = output_mode
        self.eps = float(eps)
        self.objective = objective

        self.pending_penalty_weight = float(pending_penalty_weight)
        self.pending_penalty_beta = float(pending_penalty_beta)

        self.roi_mode = roi_mode
        self.roi_combine = roi_combine
        self.roi_threshold = float(roi_threshold)
        self.roi_interval = roi_interval
        self.roi_beta = float(roi_beta)
        self.roi_bandwidth = float(roi_bandwidth)
        self.roi_min_weight = float(roi_min_weight)
        self.roi_weight_scale = float(roi_weight_scale)
        self.roi_weight_fn = roi_weight_fn

        self.noise_mode = noise_mode
        self.noise_combine = noise_combine
        self.noise_penalty_lambda = float(noise_penalty_lambda)
        self.noise_min_weight = float(noise_min_weight)
        self.noise_weight_scale = float(noise_weight_scale)
        self.noise_event_aggregate = noise_event_aggregate
        self.noise_weight_fn = noise_weight_fn

        self.noise_penalties = _expand_scalar_or_list(noise_penalty, self.m, "noise_penalty")
        if any(float(v or 0.0) != 0.0 for v in self.noise_penalties):
            self.noise_combine = "subtract"

        self.X_pending: Optional[Tensor] = None
        self.set_X_pending(X_pending)

    def set_X_pending(self, X_pending: Optional[Tensor] = None) -> None:
        self.X_pending = _coerce_reference_to_tensor(X_pending)

    def _set_eval_mode(self) -> None:
        self.model.eval()
        for sm in self.submodels:
            sm.eval()
            like = getattr(sm, "likelihood", None)
            if like is not None and hasattr(like, "eval"):
                like.eval()
            olike = getattr(sm, "ordinal_likelihood", None)
            if olike is not None and hasattr(olike, "eval"):
                olike.eval()

    def _apply_input_transform(self, X: Tensor) -> Tensor:
        X = _ensure_q_batch(X)

        it = getattr(self.model, "input_transform", None)
        if it is not None:
            Xt = it(X)
            if isinstance(Xt, tuple):
                Xt = Xt[0]
            return _ensure_q_batch(Xt)

        if self.submodels:
            it = getattr(self.submodels[0], "input_transform", None)
            if it is not None:
                Xt = it(X)
                if isinstance(Xt, tuple):
                    Xt = Xt[0]
                return _ensure_q_batch(Xt)

        return X

    def _transform_reference_like_candidate(self, X_ref, *, ref: Tensor) -> Optional[Tensor]:
        Xr = _coerce_reference_to_tensor(X_ref, ref=ref)
        if Xr is None or Xr.numel() == 0:
            return None
        Xr_t = self._apply_input_transform(_ensure_q_batch(Xr))
        return Xr_t.to(device=ref.device, dtype=ref.dtype)

    def _pending_penalty_per_point(self, Xt: Tensor) -> Tensor:
        Xt = _ensure_q_batch(Xt)
        if self.pending_penalty_weight <= 0.0:
            return Xt.new_zeros(Xt.shape[:-1])

        Xp_t = self._transform_reference_like_candidate(self.X_pending, ref=Xt)
        if Xp_t is None or Xp_t.numel() == 0:
            return Xt.new_zeros(Xt.shape[:-1])

        d = Xt.shape[-1]
        X2d = Xt.reshape(-1, d)
        Xp2d = Xp_t.reshape(-1, Xp_t.shape[-1])
        if Xp2d.shape[-1] != d:
            raise RuntimeError(
                "X_pending feature dimension mismatch after transform: "
                f"Xt.shape={tuple(Xt.shape)}, X_pending_transformed.shape={tuple(Xp_t.shape)}."
            )

        dist = torch.cdist(X2d, Xp2d).min(dim=-1).values.reshape(*Xt.shape[:-1])
        return self.pending_penalty_weight * torch.exp(-self.pending_penalty_beta * dist)

    def _summary(self, X: Tensor) -> dict[str, Tensor]:
        return stack_multi_summaries(
            self.model,
            X,
            utility_values_list=self.utility_values_list,
            noise_penalties=0.0,
            variance_scales=self.variance_scale,
            taus=self.tau,
            default_sigmas=self.default_sigma,
            eps=self.eps,
        )

    def _weights(self, ref: Tensor) -> Optional[Tensor]:
        return make_weight_tensor(self.objective_weights, ref=ref, m=self.m)

    def _aggregate_outputs(self, values: Tensor) -> Tensor:
        weights = self._weights(values)
        return aggregate_objectives(values, method=self.output_mode, weights=weights)

    def _aggregate_noise_weight(self, weight_per_output: Tensor) -> Tensor:
        weights = self._weights(weight_per_output)
        return aggregate_objectives(
            weight_per_output,
            method=self.noise_event_aggregate,
            weights=weights,
        )

    def _threshold_vector(
        self,
        values: float | Sequence[float] | Tensor,
        *,
        ref: Tensor,
        name: str,
    ) -> Tensor:
        expanded = _expand_scalar_or_list(values, self.m, name)
        return torch.as_tensor(expanded, device=ref.device, dtype=ref.dtype).view(
            *((1,) * (ref.ndim - 1)),
            self.m,
        )

    def _poe_per_output(
        self,
        summary: dict[str, Tensor],
        thresholds: float | Sequence[float] | Tensor,
    ) -> Tensor:
        thr = self._threshold_vector(
            thresholds,
            ref=summary["robust_mean"],
            name="thresholds",
        )
        z = (summary["robust_mean"] - thr) / summary["total_std"].clamp_min(self.eps)
        return _normal_cdf(z)

    def _boundary_poe_scores_for_output(
        self,
        summary: dict[str, Tensor],
        output_idx: int,
    ) -> Tensor:
        """各 ordinal cutpoint に対する P(f >= cutpoint) を返す。"""
        mean_o = summary["robust_mean"][..., output_idx]
        std_o = summary["total_std"][..., output_idx].clamp_min(self.eps)
        cp = self.cutpoints_list[output_idx].detach().to(
            device=mean_o.device,
            dtype=mean_o.dtype,
        )
        z = (mean_o.unsqueeze(-1) - cp.view(*([1] * mean_o.ndim), -1)) / std_o.unsqueeze(-1)
        return _normal_cdf(z)

    def _aggregate_boundary_outputs(
        self,
        boundary_scores_list: Sequence[Tensor],
        *,
        target_boundary_idx_list: Optional[Sequence[Optional[int]] | int] = None,
        boundary_weights_list: Optional[Sequence[Optional[Tensor | Sequence[float]]]] = None,
        boundary_reduction: BoundaryReduction = "sum",
    ) -> Tensor:
        target_list = _to_optional_list(
            target_boundary_idx_list,
            self.m,
            name="target_boundary_idx_list",
        )
        weight_list = _to_optional_list(
            boundary_weights_list,
            self.m,
            name="boundary_weights_list",
        )
        scores = []
        for o, boundary_scores in enumerate(boundary_scores_list):
            scores.append(
                _aggregate_boundary_scores(
                    boundary_scores,
                    target_boundary_idx=target_list[o],
                    boundary_weights=weight_list[o],
                    boundary_reduction=boundary_reduction,
                )
            )
        return torch.stack(scores, dim=-1)

    def _roi_thresholds_from_boundaries(
        self,
        target_boundary_idx_list: Optional[Sequence[Optional[int]] | int],
    ) -> list[Tensor]:
        """ROI の poe_* mode 用に、各出力の代表 cutpoint を 1 つ返す。

        target_boundary_idx_list が指定されている出力ではその境界を使い、
        未指定の場合は全 cutpoint の平均を使う。
        """
        target_list = _to_optional_list(
            target_boundary_idx_list,
            self.m,
            name="target_boundary_idx_list",
        )
        values: list[Tensor] = []
        for o, cp in enumerate(self.cutpoints_list):
            cp = cp.detach().reshape(-1)
            if cp.numel() == 0:
                values.append(torch.tensor(0.0, dtype=cp.dtype, device=cp.device))
                continue
            idx = target_list[o]
            if idx is None:
                values.append(cp.mean())
            else:
                idx_int = int(idx)
                if not (0 <= idx_int < cp.numel()):
                    raise ValueError(
                        f"target_boundary_idx_list[{o}] must be in [0, {cp.numel() - 1}], "
                        f"got {idx_int}."
                    )
                values.append(cp[idx_int])
        return values

    def _roi_signal(
        self,
        summary: dict[str, Tensor],
        *,
        thresholds: Optional[float | Sequence[float] | Tensor],
    ) -> Tensor:
        if self.roi_mode.startswith("poe"):
            if thresholds is None:
                raise ValueError("thresholds / targets are required for poe_* ROI modes.")
            return self._poe_per_output(summary, thresholds)
        return summary["robust_mean"]

    def _roi_weight_per_output(
        self,
        signal: Tensor,
        Xt: Tensor,
    ) -> Tensor:
        if self.roi_mode == "none":
            return torch.ones_like(signal)

        if self.roi_mode == "custom":
            if self.roi_weight_fn is None:
                raise ValueError("roi_weight_fn must be provided when roi_mode='custom'.")
            try:
                w = self.roi_weight_fn(signal, Xt)
            except TypeError:
                w = self.roi_weight_fn(signal)
            if not torch.is_tensor(w):
                raise TypeError(f"roi_weight_fn must return Tensor. Got {type(w)}.")
            return w.to(device=signal.device, dtype=signal.dtype)

        if self.roi_mode in ("mean_above", "poe_above"):
            w = torch.sigmoid(self.roi_beta * (signal - self.roi_threshold))
        elif self.roi_mode in ("mean_below", "poe_below"):
            w = torch.sigmoid(self.roi_beta * (self.roi_threshold - signal))
        elif self.roi_mode in ("mean_interval", "poe_interval"):
            if self.roi_interval is None:
                raise ValueError("roi_interval must be provided for interval ROI mode.")
            lo, hi = self.roi_interval
            w_lo = torch.sigmoid(self.roi_beta * (signal - float(lo)))
            w_hi = torch.sigmoid(self.roi_beta * (float(hi) - signal))
            w = w_lo * w_hi
        else:
            raise ValueError(f"Unknown roi_mode: {self.roi_mode!r}.")

        if self.roi_bandwidth > 0.0 and self.roi_mode in (
            "mean_above",
            "mean_below",
            "poe_above",
            "poe_below",
        ):
            band = torch.exp(-0.5 * ((signal - self.roi_threshold) / self.roi_bandwidth) ** 2)
            w = 0.5 * w + 0.5 * band

        w = self.roi_min_weight + self.roi_weight_scale * w
        return w.clamp_min(self.roi_min_weight)

    def _apply_roi_weight_per_output(
        self,
        score_per_output: Tensor,
        summary: dict[str, Tensor],
        Xt: Tensor,
        *,
        thresholds: Optional[float | Sequence[float] | Tensor],
    ) -> Tensor:
        if self.roi_mode == "none":
            return score_per_output

        signal = self._roi_signal(summary, thresholds=thresholds)
        aligned = []
        for i in range(self.m):
            si = _align_pointwise_score_to_X(
                signal[..., i],
                Xt,
                name=f"ROI signal output {i}",
                reduce_extra="mean",
            )
            aligned.append(si)
        signal = torch.stack(aligned, dim=-1)

        w = self._roi_weight_per_output(signal, Xt)
        if self.roi_combine == "multiply":
            return score_per_output * w
        if self.roi_combine == "add":
            return score_per_output + w
        raise ValueError(f"Unknown roi_combine: {self.roi_combine!r}.")

    def _noise_to_weight(self, sigma_or_var: Tensor) -> Tensor:
        v = sigma_or_var.clamp_min(0.0)

        if self.noise_mode == "none":
            w = torch.ones_like(v)
        elif self.noise_mode == "inverse_linear":
            w = 1.0 / (1.0 + self.noise_penalty_lambda * v)
        elif self.noise_mode == "inverse_sqrt":
            w = 1.0 / torch.sqrt(1.0 + self.noise_penalty_lambda * v)
        elif self.noise_mode == "inverse_exp":
            w = torch.exp(-self.noise_penalty_lambda * v)
        elif self.noise_mode == "custom":
            if self.noise_weight_fn is None:
                raise ValueError("noise_weight_fn must be provided when noise_mode='custom'.")
            try:
                w = self.noise_weight_fn(v, None)
            except TypeError:
                w = self.noise_weight_fn(v)
            if not torch.is_tensor(w):
                raise TypeError(f"noise_weight_fn must return Tensor. Got {type(w)}.")
            return w.to(device=v.device, dtype=v.dtype)
        else:
            raise ValueError(f"Unknown noise_mode: {self.noise_mode!r}.")

        if self.noise_min_weight > 0.0:
            w = self.noise_min_weight + (1.0 - self.noise_min_weight) * w
        if self.noise_weight_scale != 1.0:
            w = self.noise_weight_scale * w
        return w

    def _apply_noise_to_per_output_score(
        self,
        score_per_output: Tensor,
        summary: dict[str, Tensor],
    ) -> Tensor:
        sigma = summary.get("sigma", summary.get("total_std", None))
        if sigma is None:
            return score_per_output

        if self.noise_combine == "subtract":
            penalties = torch.as_tensor(
                [float(v or 0.0) for v in self.noise_penalties],
                device=score_per_output.device,
                dtype=score_per_output.dtype,
            )
            if penalties.abs().sum() == 0:
                penalties = torch.full_like(penalties, self.noise_penalty_lambda)
            view_shape = (1,) * (score_per_output.ndim - 1) + (self.m,)
            return score_per_output - sigma * penalties.view(*view_shape)

        if self.noise_combine == "multiply":
            if self.noise_mode == "none":
                return score_per_output
            return score_per_output * self._noise_to_weight(sigma)

        raise ValueError(f"Unknown noise_combine: {self.noise_combine!r}.")

    def _apply_objective_to_score(self, score: Tensor, X: Tensor, name: str) -> Tensor:
        return _apply_multioutput_hetero_ordinal_levelset_objective_to_score(
            self,
            score,
            X=X,
            name=name,
        )

    def _finalize_pointwise_score(
        self,
        score_per_output: Tensor,
        X: Tensor,
        *,
        summary: dict[str, Tensor],
        thresholds_for_roi: Optional[float | Sequence[float] | Tensor],
        name: str,
    ) -> Tensor:
        raw_X = _ensure_q_batch(X)
        original_batch_shape = raw_X.shape[:-2]
        Xt = self._apply_input_transform(raw_X)

        if score_per_output.shape[-1] != self.m:
            raise RuntimeError(
                f"{name}: expected score_per_output last dim {self.m}, "
                f"got shape={tuple(score_per_output.shape)}."
            )

        aligned_cols = []
        for i in range(self.m):
            si = _align_pointwise_score_to_X(
                score_per_output[..., i],
                Xt,
                name=f"{name} output {i} score",
                reduce_extra="mean",
            )
            aligned_cols.append(si)
        score_per_output = torch.stack(aligned_cols, dim=-1)

        score_per_output = self._apply_roi_weight_per_output(
            score_per_output,
            summary,
            Xt,
            thresholds=thresholds_for_roi,
        )
        score_per_output = self._apply_noise_to_per_output_score(
            score_per_output,
            summary,
        )

        score = self._aggregate_outputs(score_per_output)
        score = _align_pointwise_score_to_X(
            score,
            Xt,
            name=f"{name} aggregated score",
            reduce_extra="mean",
        )

        score = score - self._pending_penalty_per_point(Xt)

        score = _align_pointwise_score_to_X(
            score,
            Xt,
            name=f"{name} score before objective",
            reduce_extra="sum",
        )
        score = self._apply_objective_to_score(score, X=raw_X, name=name)

        out = _reduce_q(score, self.reduction)
        if out.shape != original_batch_shape:
            if out.numel() == int(torch.tensor(original_batch_shape).prod().item()):
                out = out.reshape(original_batch_shape)
            else:
                raise RuntimeError(
                    f"{name}: output shape mismatch. "
                    f"Expected {tuple(original_batch_shape)}, got {tuple(out.shape)}."
                )
        return out


# =========================================================
# Public acquisition classes: direct implementation, no aliases
# =========================================================
class qHeteroMultiOutputOrdinalProbabilityOfExceedance(
    _BaseHeteroMultiOutputOrdinalLevelSetAcquisition
):
    """heteroscedastic multi-output ordinal probability-of-exceedance acquisition。

    通常 multi-output ordinal level-set と同じく、デフォルトでは ordinal cutpoint
    ごとの boundary event ``P(f >= cutpoint_k)`` を使います。

    旧 API 互換として ``thresholds`` を渡した場合のみ、utility/robust_mean 空間の
    任意 threshold に対する PoE を計算します。
    """

    def __init__(
        self,
        model: Model,
        thresholds: Optional[float | Sequence[float] | Tensor] = None,
        *,
        boundary_weights_list: Optional[Sequence[Optional[Tensor | Sequence[float]]]] = None,
        target_boundary_idx_list: Optional[Sequence[Optional[int]] | int] = None,
        boundary_reduction: BoundaryReduction = "sum",
        **kwargs,
    ) -> None:
        super().__init__(model=model, **kwargs)
        self.thresholds = thresholds
        self.boundary_weights_list = boundary_weights_list
        self.target_boundary_idx_list = target_boundary_idx_list
        self.boundary_reduction = boundary_reduction

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._set_eval_mode()
        raw_X = _ensure_q_batch(X)
        summary = self._summary(raw_X)

        if self.thresholds is not None:
            values = self._poe_per_output(summary, self.thresholds)
            thresholds_for_roi = self.thresholds
        else:
            boundary_scores = [
                self._boundary_poe_scores_for_output(summary, o)
                for o in range(self.m)
            ]
            values = self._aggregate_boundary_outputs(
                boundary_scores,
                target_boundary_idx_list=self.target_boundary_idx_list,
                boundary_weights_list=self.boundary_weights_list,
                boundary_reduction=self.boundary_reduction,
            )
            thresholds_for_roi = self._roi_thresholds_from_boundaries(
                self.target_boundary_idx_list
            )

        return self._finalize_pointwise_score(
            values,
            raw_X,
            summary=summary,
            thresholds_for_roi=thresholds_for_roi,
            name="qHeteroMultiOutputOrdinalProbabilityOfExceedance",
        )


class qHeteroMultiOutputOrdinalLevelSetUncertainty(
    _BaseHeteroMultiOutputOrdinalLevelSetAcquisition
):
    """heteroscedastic multi-output ordinal boundary uncertainty / ICU acquisition。

    通常 multi-output ordinal の ICU と同じく、デフォルトでは各 boundary event の
    ``4 * p * (1 - p)`` を評価します。

    旧 API 互換として ``thresholds`` を渡した場合のみ、任意 threshold に対する
    ``p * (1 - p)`` を使います。
    """

    def __init__(
        self,
        model: Model,
        thresholds: Optional[float | Sequence[float] | Tensor] = None,
        *,
        boundary_weights_list: Optional[Sequence[Optional[Tensor | Sequence[float]]]] = None,
        target_boundary_idx_list: Optional[Sequence[Optional[int]] | int] = None,
        boundary_reduction: BoundaryReduction = "sum",
        **kwargs,
    ) -> None:
        super().__init__(model=model, **kwargs)
        self.thresholds = thresholds
        self.boundary_weights_list = boundary_weights_list
        self.target_boundary_idx_list = target_boundary_idx_list
        self.boundary_reduction = boundary_reduction

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._set_eval_mode()
        raw_X = _ensure_q_batch(X)
        summary = self._summary(raw_X)

        if self.thresholds is not None:
            poe = self._poe_per_output(summary, self.thresholds)
            values = poe * (1.0 - poe)
            thresholds_for_roi = self.thresholds
        else:
            boundary_scores = []
            for o in range(self.m):
                poe_b = self._boundary_poe_scores_for_output(summary, o)
                boundary_scores.append(4.0 * poe_b * (1.0 - poe_b))
            values = self._aggregate_boundary_outputs(
                boundary_scores,
                target_boundary_idx_list=self.target_boundary_idx_list,
                boundary_weights_list=self.boundary_weights_list,
                boundary_reduction=self.boundary_reduction,
            )
            thresholds_for_roi = self._roi_thresholds_from_boundaries(
                self.target_boundary_idx_list
            )

        return self._finalize_pointwise_score(
            values,
            raw_X,
            summary=summary,
            thresholds_for_roi=thresholds_for_roi,
            name="qHeteroMultiOutputOrdinalLevelSetUncertainty",
        )


class qHeteroMultiOutputOrdinalStraddle(
    _BaseHeteroMultiOutputOrdinalLevelSetAcquisition
):
    """heteroscedastic multi-output ordinal straddle acquisition。

    デフォルトでは ordinal cutpoint を boundary target として使います。
    ``targets`` を渡した場合のみ、旧 API と同じ任意 target に対する straddle になります。
    """

    def __init__(
        self,
        model: Model,
        targets: Optional[float | Sequence[float] | Tensor] = None,
        *,
        beta: float | Sequence[float] | Tensor = 1.0,
        boundary_weights_list: Optional[Sequence[Optional[Tensor | Sequence[float]]]] = None,
        target_boundary_idx_list: Optional[Sequence[Optional[int]] | int] = None,
        boundary_reduction: BoundaryReduction = "sum",
        **kwargs,
    ) -> None:
        super().__init__(model=model, **kwargs)
        self.targets = targets
        self.beta = beta
        self.boundary_weights_list = boundary_weights_list
        self.target_boundary_idx_list = target_boundary_idx_list
        self.boundary_reduction = boundary_reduction

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._set_eval_mode()
        raw_X = _ensure_q_batch(X)
        summary = self._summary(raw_X)

        beta_vec = _expand_scalar_or_list(self.beta, self.m, "beta")

        if self.targets is not None:
            targets = self._threshold_vector(
                self.targets,
                ref=summary["robust_mean"],
                name="targets",
            )
            beta = self._threshold_vector(
                self.beta,
                ref=summary["robust_mean"],
                name="beta",
            )
            values = beta * summary["total_std"] - (summary["robust_mean"] - targets).abs()
            thresholds_for_roi = self.targets
        else:
            boundary_scores = []
            for o in range(self.m):
                mean_o = summary["robust_mean"][..., o]
                std_o = summary["total_std"][..., o].clamp_min(self.eps)
                cp = self.cutpoints_list[o].detach().to(
                    device=mean_o.device,
                    dtype=mean_o.dtype,
                )
                beta_o = torch.as_tensor(
                    beta_vec[o],
                    device=mean_o.device,
                    dtype=mean_o.dtype,
                )
                score_b = beta_o * std_o.unsqueeze(-1) - (
                    mean_o.unsqueeze(-1) - cp.view(*([1] * mean_o.ndim), -1)
                ).abs()
                boundary_scores.append(score_b)

            values = self._aggregate_boundary_outputs(
                boundary_scores,
                target_boundary_idx_list=self.target_boundary_idx_list,
                boundary_weights_list=self.boundary_weights_list,
                boundary_reduction=self.boundary_reduction,
            )
            thresholds_for_roi = self._roi_thresholds_from_boundaries(
                self.target_boundary_idx_list
            )

        return self._finalize_pointwise_score(
            values,
            raw_X,
            summary=summary,
            thresholds_for_roi=thresholds_for_roi,
            name="qHeteroMultiOutputOrdinalStraddle",
        )


class qHeteroMultiOutputOrdinalBoundaryVariance(
    _BaseHeteroMultiOutputOrdinalLevelSetAcquisition
):
    """heteroscedastic multi-output ordinal boundary variance acquisition。

    デフォルトでは ordinal cutpoint 近傍の variance を評価します。
    ``targets`` を渡した場合のみ、旧 API と同じ任意 target に対する kernel variance になります。
    """

    def __init__(
        self,
        model: Model,
        targets: Optional[float | Sequence[float] | Tensor] = None,
        *,
        kernel_tau: float | Sequence[float] | Tensor = 1.0,
        boundary_weights_list: Optional[Sequence[Optional[Tensor | Sequence[float]]]] = None,
        target_boundary_idx_list: Optional[Sequence[Optional[int]] | int] = None,
        boundary_reduction: BoundaryReduction = "sum",
        **kwargs,
    ) -> None:
        super().__init__(model=model, **kwargs)
        self.targets = targets
        self.kernel_tau = kernel_tau
        self.boundary_weights_list = boundary_weights_list
        self.target_boundary_idx_list = target_boundary_idx_list
        self.boundary_reduction = boundary_reduction

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._set_eval_mode()
        raw_X = _ensure_q_batch(X)
        summary = self._summary(raw_X)

        if self.targets is not None:
            targets = self._threshold_vector(
                self.targets,
                ref=summary["robust_mean"],
                name="targets",
            )
            kernel_tau = self._threshold_vector(
                self.kernel_tau,
                ref=summary["robust_mean"],
                name="kernel_tau",
            ).clamp_min(self.eps)

            w = torch.exp(-0.5 * ((summary["robust_mean"] - targets) / kernel_tau) ** 2)
            values = summary["total_std"].pow(2) * w
            thresholds_for_roi = self.targets
        else:
            tau_vec = _expand_scalar_or_list(self.kernel_tau, self.m, "kernel_tau")
            boundary_scores = []
            for o in range(self.m):
                mean_o = summary["robust_mean"][..., o]
                std_o = summary["total_std"][..., o].clamp_min(self.eps)
                cp = self.cutpoints_list[o].detach().to(
                    device=mean_o.device,
                    dtype=mean_o.dtype,
                )
                tau_o = torch.as_tensor(
                    tau_vec[o],
                    device=mean_o.device,
                    dtype=mean_o.dtype,
                ).clamp_min(self.eps)
                w_b = _boundary_kernel_scores(mean_o, cp, tau=tau_o)
                boundary_scores.append(std_o.pow(2).unsqueeze(-1) * w_b)

            values = self._aggregate_boundary_outputs(
                boundary_scores,
                target_boundary_idx_list=self.target_boundary_idx_list,
                boundary_weights_list=self.boundary_weights_list,
                boundary_reduction=self.boundary_reduction,
            )
            thresholds_for_roi = self._roi_thresholds_from_boundaries(
                self.target_boundary_idx_list
            )

        return self._finalize_pointwise_score(
            values,
            raw_X,
            summary=summary,
            thresholds_for_roi=thresholds_for_roi,
            name="qHeteroMultiOutputOrdinalBoundaryVariance",
        )


__all__ = [
    "HeteroMultiOutputOrdinalLevelSetScoreObjective",
    "_MultiObjectiveHeteroOrdinalLevelSetScoreObjective",
    "qHeteroMultiOutputOrdinalProbabilityOfExceedance",
    "qHeteroMultiOutputOrdinalLevelSetUncertainty",
    "qHeteroMultiOutputOrdinalStraddle",
    "qHeteroMultiOutputOrdinalBoundaryVariance",
]
