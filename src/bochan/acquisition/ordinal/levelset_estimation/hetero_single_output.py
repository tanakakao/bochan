from __future__ import annotations

import math
from typing import Callable, Literal, Optional, Sequence

import torch
from torch import Tensor

from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.models.model import Model
from botorch.utils.transforms import t_batch_mode_transform

from ..hetero_utils import _normal_cdf, get_hetero_ordinal_summary


RiskType = Optional[Literal["var", "cvar"]]
ReductionType = Literal["mean", "sum", "max"]
NoiseWeightMode = Literal["none", "inverse_linear", "inverse_exp", "custom"]
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
BoundaryReduction = Literal["sum", "mean", "max", "min"]


# =========================================================
# Score objective
# =========================================================
class HeteroOrdinalLevelSetScoreObjective(torch.nn.Module):
    """
    hetero ordinal level-set acquisition の pointwise score に作用する objective。

    posterior samples ではなく、level-set acquisition 内で計算済みの score に作用する。
    主な用途は InputPerturbation による q * n_w を q に戻すこと。
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
                    "HeteroOrdinalLevelSetScoreObjective received an aggregated score. "
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



# =========================================================
# Helpers
# =========================================================
def _ensure_q_batch(X: Tensor) -> Tensor:
    if not torch.is_tensor(X):
        raise TypeError(f"X must be a Tensor. Got {type(X)}.")
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


def _prepare_boundary_weights(
    boundary_weights: Optional[Tensor | Sequence[float]],
    n_boundaries: int,
    *,
    device,
    dtype,
) -> Tensor:
    """境界ごとの重みを Tensor に正規化する。"""
    if boundary_weights is None:
        return torch.ones(n_boundaries, device=device, dtype=dtype)

    w = torch.as_tensor(boundary_weights, device=device, dtype=dtype).detach().reshape(-1)
    if w.numel() != n_boundaries:
        raise ValueError(
            f"boundary_weights must have length {n_boundaries}, got {w.numel()}."
        )
    return w


def _validate_target_boundary_idx(
    target_boundary_idx: Optional[int],
    n_boundaries: int,
) -> Optional[int]:
    """target_boundary_idx が境界数の範囲内か確認する。"""
    if target_boundary_idx is None:
        return None

    idx = int(target_boundary_idx)
    if not (0 <= idx < n_boundaries):
        raise ValueError(
            f"target_boundary_idx must satisfy 0 <= idx < {n_boundaries}. "
            f"Got {target_boundary_idx}."
        )
    return idx


def _aggregate_boundary_scores(
    boundary_scores: Tensor,
    *,
    target_boundary_idx: Optional[int] = None,
    boundary_weights: Optional[Tensor | Sequence[float]] = None,
    boundary_reduction: BoundaryReduction = "sum",
) -> Tensor:
    """境界ごとの score を pointwise score に集約する。

    ``target_boundary_idx=k`` は class k / class k+1 の境界を意味する。
    例えば 3 クラス 0/1/2 では、0 が 0/1 境界、1 が 1/2 境界。
    """
    if boundary_scores.ndim < 1:
        raise RuntimeError("boundary_scores must have a boundary dimension.")

    n_boundaries = boundary_scores.shape[-1]
    idx = _validate_target_boundary_idx(target_boundary_idx, n_boundaries)
    if idx is not None:
        return boundary_scores[..., idx]

    if boundary_weights is not None:
        w = _prepare_boundary_weights(
            boundary_weights,
            n_boundaries=n_boundaries,
            device=boundary_scores.device,
            dtype=boundary_scores.dtype,
        )
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


def _try_call_zero_arg(obj):
    """callable な属性なら zero-arg call し、そうでなければそのまま返す。"""
    return obj() if callable(obj) else obj


def _get_ordinal_likelihood(model: Optional[Model]):
    """model から ordinal likelihood 相当のオブジェクトを取得する。"""
    if model is None:
        return None
    if hasattr(model, "ordinal_likelihood"):
        return getattr(model, "ordinal_likelihood")
    if hasattr(model, "likelihood"):
        return getattr(model, "likelihood")
    return None


def _get_cutpoints_from_likelihood(ordinal_likelihood) -> Tensor:
    """ordinal likelihood から cutpoints / thresholds を取得する。

    通常 ordinal 側の実装と同様に、複数の属性名を許容する。
    ここでは ``utility_values=None`` のときのクラス数推定に使うため、
    取得した cutpoints は detach して定数として扱う。
    """
    if ordinal_likelihood is None:
        raise ValueError("ordinal_likelihood must not be None.")

    if hasattr(ordinal_likelihood, "get_cutpoints"):
        cutpoints = _try_call_zero_arg(getattr(ordinal_likelihood, "get_cutpoints"))
        return torch.as_tensor(cutpoints).detach().clone().reshape(-1)

    for name in ("transformed_cutpoints", "cutpoints", "thresholds", "cuts", "cutoffs"):
        if hasattr(ordinal_likelihood, name):
            cutpoints = _try_call_zero_arg(getattr(ordinal_likelihood, name))
            return torch.as_tensor(cutpoints).detach().clone().reshape(-1)

    if hasattr(ordinal_likelihood, "raw_cutpoints"):
        raw = _try_call_zero_arg(getattr(ordinal_likelihood, "raw_cutpoints"))
        raw = torch.as_tensor(raw).detach().clone()
        if hasattr(ordinal_likelihood, "transform_cutpoints"):
            cutpoints = ordinal_likelihood.transform_cutpoints(raw)
            return torch.as_tensor(cutpoints).detach().clone().reshape(-1)
        return raw.detach().clone().reshape(-1)

    raise ValueError(
        "Could not find cutpoints on ordinal likelihood. "
        "Expected one of: get_cutpoints / transformed_cutpoints / cutpoints / "
        "thresholds / cuts / cutoffs / raw_cutpoints."
    )


def _default_ordinal_utility_values_from_model(
    model: Optional[Model],
    *,
    reference: Tensor,
) -> Tensor:
    """cutpoints 数から default utility [0, 1, ..., K-1] を生成する。"""
    ordinal_likelihood = _get_ordinal_likelihood(model)
    if ordinal_likelihood is None:
        raise ValueError(
            "utility_values is None, so target_boundary_idx / boundary aggregation "
            "requires model.ordinal_likelihood or model.likelihood with cutpoints."
        )

    cutpoints = _get_cutpoints_from_likelihood(ordinal_likelihood)
    n_classes = int(cutpoints.numel()) + 1
    if n_classes < 2:
        raise ValueError(
            "At least two ordinal classes are required. "
            f"Inferred n_classes={n_classes} from cutpoints."
        )

    return torch.arange(n_classes, device=reference.device, dtype=reference.dtype)


def _utility_boundary_values(
    utility_values: Optional[Sequence[float] | Tensor],
    *,
    reference: Tensor,
    model: Optional[Model] = None,
) -> Tensor:
    """utility scale 上の ordinal 境界値を返す。

    hetero ordinal の level-set score は ``summary["robust_mean"]``、つまり
    utility scale 上の平均に対して計算される。そのため、通常 ordinal の
    latent cutpoint ではなく、隣接クラス utility の中点を境界として使う。

    ``utility_values`` が None の場合は、通常 ordinal と同じ使用感にするため、
    model の cutpoints 数からクラス数 K を推定し、
    ``[0, 1, ..., K - 1]`` を default utility として自動生成する。
    """
    if utility_values is None:
        u = _default_ordinal_utility_values_from_model(model, reference=reference)
    else:
        u = torch.as_tensor(utility_values, device=reference.device, dtype=reference.dtype).reshape(-1)

    if u.numel() < 2:
        raise ValueError("utility_values must contain at least two class utilities.")

    return 0.5 * (u[:-1] + u[1:])


def _representative_boundary_target(
    boundary_targets: Tensor,
    *,
    target_boundary_idx: Optional[int],
    boundary_weights: Optional[Tensor | Sequence[float]] = None,
) -> Tensor:
    """ROI 用の代表境界値を返す。"""
    idx = _validate_target_boundary_idx(target_boundary_idx, boundary_targets.numel())
    if idx is not None:
        return boundary_targets[idx]

    if boundary_weights is None:
        return boundary_targets.mean()

    w = _prepare_boundary_weights(
        boundary_weights,
        n_boundaries=boundary_targets.numel(),
        device=boundary_targets.device,
        dtype=boundary_targets.dtype,
    )
    return (boundary_targets * w).sum() / w.sum().clamp_min(1e-12)


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

    Fully Bayesian / SAAS / MC extra dim などが前に乗る場合は平均または和で潰す。
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


def _apply_hetero_ordinal_levelset_objective_to_score(
    owner,
    score: Tensor,
    X: Optional[Tensor] = None,
    name: str = "HeteroOrdinalLevelSet",
) -> Tensor:
    objective = getattr(owner, "objective", None)
    if objective is None:
        return score

    out = _objective_call(objective, score, X)
    if not torch.is_tensor(out):
        raise RuntimeError(f"{name}: objective must return a Tensor. Got {type(out)}.")
    return out




# =========================================================
# Classification-aligned hetero ordinal level-set base
# =========================================================
def _bernoulli_boundary_uncertainty(prob: Tensor) -> Tensor:
    """P(y >= boundary) に対する binary boundary uncertainty を返す。"""
    return 4.0 * prob * (1.0 - prob)


def _entropy_from_probs(probs: Tensor, eps: float) -> Tensor:
    """class probability の Shannon entropy を返す。"""
    p = probs.clamp_min(eps)
    p = p / p.sum(dim=-1, keepdim=True).clamp_min(eps)
    return -(p * p.log()).sum(dim=-1)


def _sigmoid(x: Tensor) -> Tensor:
    return torch.sigmoid(x)


def _ordinal_class_probs_from_f(f: Tensor, ordinal_likelihood) -> Tensor:
    """latent f から ordinal class probability を計算する。"""
    for name in (
        "class_probs_from_f",
        "probs_from_f",
        "predict_proba_from_f",
        "class_probabilities_from_f",
        "marginal_probs_from_f",
    ):
        if hasattr(ordinal_likelihood, name):
            probs = getattr(ordinal_likelihood, name)(f)
            if hasattr(probs, "probs"):
                probs = probs.probs
            probs = torch.as_tensor(probs, device=f.device, dtype=f.dtype)
            probs = probs.clamp_min(1e-12)
            return probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    cutpoints = _get_cutpoints_from_likelihood(ordinal_likelihood).to(
        device=f.device,
        dtype=f.dtype,
    )
    z = cutpoints.view(*([1] * f.ndim), -1) - f.unsqueeze(-1)
    cdf = _sigmoid(z)
    p0 = cdf[..., :1]
    if cutpoints.numel() > 1:
        pmid = cdf[..., 1:] - cdf[..., :-1]
        plast = 1.0 - cdf[..., -1:]
        probs = torch.cat([p0, pmid, plast], dim=-1)
    else:
        plast = 1.0 - cdf[..., -1:]
        probs = torch.cat([p0, plast], dim=-1)
    probs = probs.clamp_min(1e-12)
    return probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def _align_class_probs_to_X(probs: Tensor, Xt: Tensor, *, name: str) -> Tensor:
    """class probability を Xt.shape[:-1] + (K,) に揃える。"""
    if not torch.is_tensor(probs):
        raise TypeError(f"{name} must be a Tensor. Got {type(probs)}.")
    if probs.ndim < 2:
        raise RuntimeError(f"{name}: class probs must have at least 2 dims. Got {tuple(probs.shape)}.")

    target_prefix = Xt.shape[:-1]
    k = probs.shape[-1]
    target = target_prefix + torch.Size([k])
    out = probs

    if out.shape == target:
        return out

    # posterior.mean が (..., q, K, 1) のような形の場合に対応。
    if out.ndim >= 1 and out.shape[-1] == 1 and k == 1:
        squeezed = out.squeeze(-1)
        if squeezed.ndim >= 2:
            out = squeezed
            k = out.shape[-1]
            target = target_prefix + torch.Size([k])
            if out.shape == target:
                return out

    # 総要素数が一致する場合は reshape でそろえる。
    expected_numel = int(torch.tensor(target).prod().item())
    if out.numel() == expected_numel:
        return out.reshape(target)

    # Fully Bayesian / ensemble / MC extra dim は平均で潰す。
    while out.ndim > len(target):
        out = out.mean(dim=0)
        if out.shape == target:
            return out

    if out.shape == target:
        return out

    # prefix が broadcast 可能な場合。
    if out.ndim == len(target) and out.shape[-1] == target[-1]:
        try:
            return out.expand(target)
        except RuntimeError:
            pass

    raise RuntimeError(
        f"{name}: class probability shape mismatch. "
        f"probs.shape={tuple(probs.shape)}, expected={tuple(target)}, Xt.shape={tuple(Xt.shape)}."
    )


class _BaseHeteroOrdinalLevelSetAcquisition(AcquisitionFunction):
    """heteroscedastic ordinal level-set acquisition の共通基底。

    classification の hetero binary level-set acquisition と API / 処理順序を
    できるだけ揃えています。

    Standard order:
        pointwise score
        -> ROI weighting
        -> noise weighting / penalty
        -> pending penalty
        -> objective
        -> q reduction
    """

    def __init__(
        self,
        model: Model,
        reduction: ReductionType = "mean",
        pending_penalty_weight: float = 0.0,
        pending_penalty_beta: float = 10.0,
        eps: float = 1e-6,
        # ROI
        roi_mode: ROIWeightMode = "none",
        roi_combine: ROICombineType = "multiply",
        roi_threshold: float = 0.5,
        roi_target_prob: float = 0.8,
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
        noise_model_outputs_log_var: bool = True,
        noise_weight_fn: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
        # objective
        objective: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
        # ordinal / hetero summary
        utility_values: Optional[Sequence[float] | Tensor] = None,
        variance_scale: float = 1.0,
        summary_tau: float = 1e-6,
        default_sigma: float = 0.0,
        X_pending: Optional[Tensor] = None,
    ) -> None:
        super().__init__(model=model)

        if reduction not in ("mean", "sum", "max"):
            raise ValueError("reduction must be 'mean', 'sum', or 'max'.")
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
        if noise_mode not in ("none", "inverse_linear", "inverse_exp", "custom"):
            raise ValueError(f"Unknown noise_mode: {noise_mode!r}.")
        if noise_combine not in ("multiply", "subtract"):
            raise ValueError("noise_combine must be 'multiply' or 'subtract'.")

        self.reduction = reduction
        self.pending_penalty_weight = float(pending_penalty_weight)
        self.pending_penalty_beta = float(pending_penalty_beta)
        self.eps = float(eps)

        self.roi_mode = roi_mode
        self.roi_combine = roi_combine
        self.roi_threshold = float(roi_threshold)
        self.roi_target_prob = float(roi_target_prob)
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
        self.noise_model_outputs_log_var = bool(noise_model_outputs_log_var)
        self.noise_weight_fn = noise_weight_fn

        self.objective = objective
        self.utility_values = utility_values
        self.variance_scale = float(variance_scale)
        self.summary_tau = float(summary_tau)
        self.default_sigma = float(default_sigma)

        self.X_pending: Optional[Tensor] = None
        self.set_X_pending(X_pending)

    def set_X_pending(self, X_pending: Optional[Tensor] = None) -> None:
        """BoTorch optimize_acqf(sequential=True) などから pending 点を更新する。"""
        self.X_pending = _coerce_reference_to_tensor(X_pending)

    def _prepare_eval(self) -> None:
        self.model.eval()
        for attr in ("likelihood", "ordinal_likelihood"):
            obj = getattr(self.model, attr, None)
            if obj is not None and hasattr(obj, "eval"):
                obj.eval()

    def _ensure_q_batch(self, X: Tensor) -> Tensor:
        return _ensure_q_batch(X)

    def _apply_input_transform(self, X: Tensor) -> Tensor:
        X = _ensure_q_batch(X)
        it = getattr(self.model, "input_transform", None)
        if it is not None:
            Xt = it(X)
            if isinstance(Xt, tuple):
                Xt = Xt[0]
            return _ensure_q_batch(Xt)

        models = getattr(self.model, "models", None)
        if models is not None and len(models) > 0:
            it = getattr(models[0], "input_transform", None)
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
        return get_hetero_ordinal_summary(
            self.model,
            X,
            utility_values=self.utility_values,
            noise_penalty=0.0,
            variance_scale=self.variance_scale,
            tau=self.summary_tau,
            default_sigma=self.default_sigma,
            eps=self.eps,
        )

    def _threshold_like(self, value: float | Tensor, reference: Tensor) -> Tensor:
        return torch.as_tensor(value, device=reference.device, dtype=reference.dtype)

    def _boundary_targets(self, reference: Tensor) -> Tensor:
        return _utility_boundary_values(
            self.utility_values,
            reference=reference,
            model=self.model,
        )

    def _representative_boundary_target(
        self,
        boundary_targets: Tensor,
        *,
        target_boundary_idx: Optional[int],
        boundary_weights: Optional[Tensor | Sequence[float]],
    ) -> Tensor:
        return _representative_boundary_target(
            boundary_targets,
            target_boundary_idx=target_boundary_idx,
            boundary_weights=boundary_weights,
        )

    def _boundary_exceedance_probs(
        self,
        summary: dict[str, Tensor],
        boundary_targets: Tensor,
    ) -> Tensor:
        mean = summary["robust_mean"]
        total_std = summary["total_std"].clamp_min(self.eps)
        th = boundary_targets.view(*([1] * mean.ndim), -1)
        z = (mean.unsqueeze(-1) - th) / total_std.unsqueeze(-1)
        return _normal_cdf(z)

    def _compute_poe(self, summary: dict[str, Tensor], threshold: float | Tensor) -> Tensor:
        th = self._threshold_like(threshold, summary["robust_mean"])
        z = (summary["robust_mean"] - th) / summary["total_std"].clamp_min(self.eps)
        return _normal_cdf(z)

    def _roi_weight_per_point(self, signal: Tensor, X: Optional[Tensor]) -> Tensor:
        if self.roi_mode == "none":
            return torch.ones_like(signal)

        if self.roi_mode == "custom":
            if self.roi_weight_fn is None:
                raise ValueError("roi_weight_fn must be provided when roi_mode='custom'.")
            try:
                w = self.roi_weight_fn(signal, X)
            except TypeError:
                w = self.roi_weight_fn(signal)
            if not torch.is_tensor(w):
                raise TypeError(f"roi_weight_fn must return Tensor. Got {type(w)}.")
            return w.to(device=signal.device, dtype=signal.dtype)

        target = self.roi_target_prob if self.roi_mode.startswith("poe") else self.roi_threshold

        if self.roi_mode in ("mean_above", "poe_above"):
            w = torch.sigmoid(self.roi_beta * (signal - target))
        elif self.roi_mode in ("mean_below", "poe_below"):
            w = torch.sigmoid(self.roi_beta * (target - signal))
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
            band = torch.exp(-0.5 * ((signal - target) / self.roi_bandwidth) ** 2)
            w = 0.5 * w + 0.5 * band

        w = self.roi_min_weight + self.roi_weight_scale * w
        return w.clamp_min(self.roi_min_weight)

    def _apply_roi_weight_per_point(self, score: Tensor, signal: Tensor, Xt: Tensor) -> Tensor:
        if self.roi_mode == "none":
            return score

        signal = _align_pointwise_score_to_X(
            signal,
            Xt,
            name="ROI signal",
            reduce_extra="mean",
        )
        w = self._roi_weight_per_point(signal, Xt)
        if w.shape != score.shape:
            w = _align_pointwise_score_to_X(
                w,
                Xt,
                name="ROI weight",
                reduce_extra="mean",
            )

        if self.roi_combine == "multiply":
            return score * w
        if self.roi_combine == "add":
            return score + w
        raise ValueError(f"Unknown roi_combine: {self.roi_combine!r}.")

    def _noise_sigma_from_summary(self, summary: dict[str, Tensor], Xt: Tensor) -> Tensor:
        sigma = summary.get("sigma", None)
        if sigma is None:
            sigma = summary.get("noise_std", None)
        if sigma is None:
            sigma = summary.get("total_std", None)
        if sigma is None:
            return Xt.new_zeros(Xt.shape[:-1])
        return _align_pointwise_score_to_X(
            sigma,
            Xt,
            name="noise sigma",
            reduce_extra="mean",
        ).clamp_min(0.0)

    def _noise_weight_per_point(self, sigma: Tensor, X: Optional[Tensor]) -> Tensor:
        sigma = sigma.clamp_min(0.0)

        if self.noise_mode == "none":
            return torch.ones_like(sigma)

        if self.noise_mode == "custom":
            if self.noise_weight_fn is None:
                raise ValueError("noise_weight_fn must be provided when noise_mode='custom'.")
            try:
                w = self.noise_weight_fn(sigma, X)
            except TypeError:
                w = self.noise_weight_fn(sigma)
            if not torch.is_tensor(w):
                raise TypeError(f"noise_weight_fn must return Tensor. Got {type(w)}.")
            return w.to(device=sigma.device, dtype=sigma.dtype)

        if self.noise_mode == "inverse_linear":
            w = 1.0 / (1.0 + self.noise_penalty_lambda * sigma)
        elif self.noise_mode == "inverse_exp":
            w = torch.exp(-self.noise_penalty_lambda * sigma)
        else:
            raise ValueError(f"Unknown noise_mode: {self.noise_mode!r}.")

        w = self.noise_min_weight + self.noise_weight_scale * w
        return w.clamp_min(self.noise_min_weight)

    def _apply_noise_weight_per_point(self, score: Tensor, summary: dict[str, Tensor], Xt: Tensor) -> Tensor:
        sigma = self._noise_sigma_from_summary(summary, Xt)

        if self.noise_combine == "subtract":
            return score - self.noise_penalty_lambda * sigma

        if self.noise_combine == "multiply":
            if self.noise_mode == "none":
                return score
            return score * self._noise_weight_per_point(sigma, Xt)

        raise ValueError(f"Unknown noise_combine: {self.noise_combine!r}.")

    def _apply_score_objective(self, score: Tensor, X: Optional[Tensor], *, name: str) -> Tensor:
        return _apply_hetero_ordinal_levelset_objective_to_score(self, score, X=X, name=name)

    def _roi_signal_from_summary(
        self,
        summary: dict[str, Tensor],
        *,
        threshold_for_roi: Optional[float | Tensor],
    ) -> Tensor:
        if self.roi_mode.startswith("poe"):
            if threshold_for_roi is None:
                boundary_targets = self._boundary_targets(summary["robust_mean"])
                threshold_for_roi = boundary_targets.mean()
            return self._compute_poe(summary, threshold_for_roi)
        return summary["robust_mean"]

    def _postprocess_pointwise_score(
        self,
        score: Tensor,
        summary: dict[str, Tensor],
        Xt: Tensor,
        X: Tensor,
        *,
        name: str,
        threshold_for_roi: Optional[float | Tensor] = None,
    ) -> Tensor:
        score = _align_pointwise_score_to_X(
            score,
            Xt,
            name=f"{name} score before weighting",
            reduce_extra="mean",
        )

        roi_signal = self._roi_signal_from_summary(summary, threshold_for_roi=threshold_for_roi)
        score = self._apply_roi_weight_per_point(score, roi_signal, Xt)
        score = self._apply_noise_weight_per_point(score, summary, Xt)

        pending = self._pending_penalty_per_point(Xt)
        if pending.shape == score.shape:
            score = score - pending
        elif pending.numel() == score.numel():
            score = score - pending.reshape_as(score)
        elif self.pending_penalty_weight > 0.0:
            raise RuntimeError(
                f"Pending penalty shape mismatch in {name}: "
                f"score.shape={tuple(score.shape)}, pending.shape={tuple(pending.shape)}"
            )

        score = _align_pointwise_score_to_X(
            score,
            Xt,
            name=f"{name} score before objective",
            reduce_extra="sum",
        )
        return self._apply_score_objective(score, X=X, name=name)

    def _reduce_q(self, score: Tensor) -> Tensor:
        return _reduce_q(score, self.reduction)

    def _check_output_shape(self, out: Tensor, original_batch_shape: torch.Size, name: str) -> None:
        if out.shape == original_batch_shape:
            return
        if out.numel() == int(torch.tensor(original_batch_shape).prod().item()):
            return
        raise RuntimeError(
            f"{name}: output shape mismatch. "
            f"Expected {tuple(original_batch_shape)}, got {tuple(out.shape)}."
        )

    def _maybe_reshape_output(self, out: Tensor, original_batch_shape: torch.Size) -> Tensor:
        if out.shape == original_batch_shape:
            return out
        if out.numel() == int(torch.tensor(original_batch_shape).prod().item()):
            return out.reshape(original_batch_shape)
        return out

    def _get_latent_posterior(self, X: Tensor):
        for name in ("latent_posterior", "posterior_latent", "posterior_f"):
            fn = getattr(self.model, name, None)
            if callable(fn):
                return fn(X)

        inner_model = getattr(self.model, "model", None)
        if inner_model is not None and callable(getattr(inner_model, "posterior", None)):
            return inner_model.posterior(X)

        gp_model = getattr(self.model, "gp_model", None)
        if gp_model is not None and callable(getattr(gp_model, "posterior", None)):
            return gp_model.posterior(X)

        return None

    def _class_probs_from_summary_or_model(
        self,
        X: Tensor,
        summary: dict[str, Tensor],
        Xt: Tensor,
    ) -> Tensor:
        for key in (
            "class_probs",
            "probs",
            "probabilities",
            "predictive_probs",
            "mean_probs",
            "mean_class_probs",
            "posterior_probs",
        ):
            probs = summary.get(key, None)
            if probs is not None:
                probs = _align_class_probs_to_X(probs, Xt, name=f"summary[{key!r}]")
                return probs.clamp_min(self.eps) / probs.sum(dim=-1, keepdim=True).clamp_min(self.eps)

        for name in ("predict_proba", "predict_probs", "class_probs", "predict_class_probs"):
            fn = getattr(self.model, name, None)
            if callable(fn):
                probs = fn(X)
                if hasattr(probs, "probs"):
                    probs = probs.probs
                probs = _align_class_probs_to_X(torch.as_tensor(probs, device=Xt.device, dtype=Xt.dtype), Xt, name=name)
                return probs.clamp_min(self.eps) / probs.sum(dim=-1, keepdim=True).clamp_min(self.eps)

        posterior = self.model.posterior(X)
        for attr in ("probs", "probabilities"):
            probs = getattr(posterior, attr, None)
            if probs is not None:
                probs = _align_class_probs_to_X(torch.as_tensor(probs, device=Xt.device, dtype=Xt.dtype), Xt, name=f"posterior.{attr}")
                return probs.clamp_min(self.eps) / probs.sum(dim=-1, keepdim=True).clamp_min(self.eps)

        mean = getattr(posterior, "mean", None)
        if mean is not None and torch.is_tensor(mean) and mean.ndim >= 2 and mean.shape[-1] > 1:
            probs = _align_class_probs_to_X(mean, Xt, name="posterior.mean")
            if probs.min() < 0.0 or probs.max() > 1.0:
                probs = probs.softmax(dim=-1)
            return probs.clamp_min(self.eps) / probs.sum(dim=-1, keepdim=True).clamp_min(self.eps)

        ordinal_likelihood = _get_ordinal_likelihood(self.model)
        latent_post = self._get_latent_posterior(X)
        if ordinal_likelihood is not None and latent_post is not None:
            f_mean = getattr(latent_post, "mean", None)
            if f_mean is not None:
                f_mean = _align_pointwise_score_to_X(f_mean, Xt, name="latent mean", reduce_extra="mean")
                probs = _ordinal_class_probs_from_f(f_mean, ordinal_likelihood)
                probs = _align_class_probs_to_X(probs, Xt, name="ordinal probs from latent mean")
                return probs.clamp_min(self.eps) / probs.sum(dim=-1, keepdim=True).clamp_min(self.eps)

        raise RuntimeError(
            "Could not obtain ordinal class probabilities for qHeteroOrdinalClassEntropyAcquisition. "
            "Expected get_hetero_ordinal_summary to return class_probs/probs, model to expose "
            "predict_proba/class_probs, posterior.probs, posterior.mean with class dimension, "
            "or latent_posterior plus ordinal_likelihood."
        )


# =========================================================
# Classification-aligned acquisition implementations
# =========================================================
class qHeteroOrdinalLatentStraddleAcquisition(_BaseHeteroOrdinalLevelSetAcquisition):
    """heteroscedastic ordinal 用 straddle acquisition。

    classification の qHeteroBinaryLatentStraddleAcquisition に対応します。
    ordinal では ``target_boundary_idx`` で class k / class k+1 境界を指定できます。
    ``threshold`` を直接指定した場合は utility scale 上の単一境界として扱います。
    """

    def __init__(
        self,
        model: Model,
        beta: float = 2.0,
        threshold: Optional[float | Tensor] = None,
        *,
        target_boundary_idx: Optional[int] = None,
        boundary_weights: Optional[Tensor | Sequence[float]] = None,
        boundary_reduction: BoundaryReduction = "max",
        **kwargs,
    ) -> None:
        super().__init__(model=model, **kwargs)
        if threshold is not None and target_boundary_idx is not None:
            raise ValueError("Specify either threshold or target_boundary_idx, not both.")
        self.beta = float(beta)
        self.threshold = threshold
        self.target_boundary_idx = target_boundary_idx
        self.boundary_weights = boundary_weights
        self.boundary_reduction = boundary_reduction

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._prepare_eval()
        X_in = _ensure_q_batch(X)
        original_batch_shape = X_in.shape[:-2]
        Xt = self._apply_input_transform(X_in)
        summary = self._summary(X_in)

        mean = summary["robust_mean"]
        std = summary["total_std"].clamp_min(self.eps)

        if self.threshold is not None:
            threshold = self._threshold_like(self.threshold, mean)
            score = (self.beta ** 0.5) * std - (mean - threshold).abs()
            threshold_for_roi = self.threshold
        else:
            boundary_targets = self._boundary_targets(mean)
            dist_b = (mean.unsqueeze(-1) - boundary_targets.view(*([1] * mean.ndim), -1)).abs()
            score_b = (self.beta ** 0.5) * std.unsqueeze(-1) - dist_b
            score = _aggregate_boundary_scores(
                score_b,
                target_boundary_idx=self.target_boundary_idx,
                boundary_weights=self.boundary_weights,
                boundary_reduction=self.boundary_reduction,
            )
            threshold_for_roi = self._representative_boundary_target(
                boundary_targets,
                target_boundary_idx=self.target_boundary_idx,
                boundary_weights=self.boundary_weights,
            )

        score = self._postprocess_pointwise_score(
            score,
            summary,
            Xt,
            X_in,
            name="qHeteroOrdinalLatentStraddleAcquisition",
            threshold_for_roi=threshold_for_roi,
        )
        out = self._maybe_reshape_output(self._reduce_q(score), original_batch_shape)
        self._check_output_shape(out, original_batch_shape, "qHeteroOrdinalLatentStraddleAcquisition")
        return out


class qHeteroOrdinalICUAcquisition(_BaseHeteroOrdinalLevelSetAcquisition):
    """heteroscedastic ordinal 用 ICU acquisition。

    classification の qHeteroBinaryICUAcquisition に対応します。
    ordinal では各 class 境界の ``P(y >= boundary) * (1 - P(y >= boundary))`` を評価し、
    ``target_boundary_idx`` / ``boundary_weights`` / ``boundary_reduction`` で境界方向を集約します。
    """

    def __init__(
        self,
        model: Model,
        *,
        target_boundary_idx: Optional[int] = None,
        boundary_weights: Optional[Tensor | Sequence[float]] = None,
        boundary_reduction: BoundaryReduction = "sum",
        **kwargs,
    ) -> None:
        super().__init__(model=model, **kwargs)
        self.target_boundary_idx = target_boundary_idx
        self.boundary_weights = boundary_weights
        self.boundary_reduction = boundary_reduction

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._prepare_eval()
        X_in = _ensure_q_batch(X)
        original_batch_shape = X_in.shape[:-2]
        Xt = self._apply_input_transform(X_in)
        summary = self._summary(X_in)

        boundary_targets = self._boundary_targets(summary["robust_mean"])
        poe_b = self._boundary_exceedance_probs(summary, boundary_targets)
        score_b = _bernoulli_boundary_uncertainty(poe_b)
        score = _aggregate_boundary_scores(
            score_b,
            target_boundary_idx=self.target_boundary_idx,
            boundary_weights=self.boundary_weights,
            boundary_reduction=self.boundary_reduction,
        )
        threshold_for_roi = self._representative_boundary_target(
            boundary_targets,
            target_boundary_idx=self.target_boundary_idx,
            boundary_weights=self.boundary_weights,
        )

        score = self._postprocess_pointwise_score(
            score,
            summary,
            Xt,
            X_in,
            name="qHeteroOrdinalICUAcquisition",
            threshold_for_roi=threshold_for_roi,
        )
        out = self._maybe_reshape_output(self._reduce_q(score), original_batch_shape)
        self._check_output_shape(out, original_batch_shape, "qHeteroOrdinalICUAcquisition")
        return out


class qHeteroOrdinalBoundaryVarianceAcquisition(_BaseHeteroOrdinalLevelSetAcquisition):
    """heteroscedastic ordinal 用 boundary variance acquisition。

    classification の qHeteroBinaryBoundaryVarianceAcquisition に対応します。
    ``tau`` は境界近傍 kernel の幅です。
    """

    def __init__(
        self,
        model: Model,
        threshold: Optional[float | Tensor] = None,
        tau: float = 1.0,
        *,
        target_boundary_idx: Optional[int] = None,
        boundary_weights: Optional[Tensor | Sequence[float]] = None,
        boundary_reduction: BoundaryReduction = "sum",
        **kwargs,
    ) -> None:
        super().__init__(model=model, **kwargs)
        if threshold is not None and target_boundary_idx is not None:
            raise ValueError("Specify either threshold or target_boundary_idx, not both.")
        self.threshold = threshold
        self.tau = float(tau)
        self.target_boundary_idx = target_boundary_idx
        self.boundary_weights = boundary_weights
        self.boundary_reduction = boundary_reduction

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._prepare_eval()
        X_in = _ensure_q_batch(X)
        original_batch_shape = X_in.shape[:-2]
        Xt = self._apply_input_transform(X_in)
        summary = self._summary(X_in)

        mean = summary["robust_mean"]
        total_var = summary["total_std"].pow(2)
        tau = torch.as_tensor(self.tau, device=mean.device, dtype=mean.dtype).clamp_min(self.eps)

        if self.threshold is not None:
            threshold = self._threshold_like(self.threshold, mean)
            weight = torch.exp(-0.5 * ((mean - threshold) / tau) ** 2)
            score = total_var * weight
            threshold_for_roi = self.threshold
        else:
            boundary_targets = self._boundary_targets(mean)
            z_b = (mean.unsqueeze(-1) - boundary_targets.view(*([1] * mean.ndim), -1)) / tau
            weight_b = torch.exp(-0.5 * z_b.pow(2))
            score_b = total_var.unsqueeze(-1) * weight_b
            score = _aggregate_boundary_scores(
                score_b,
                target_boundary_idx=self.target_boundary_idx,
                boundary_weights=self.boundary_weights,
                boundary_reduction=self.boundary_reduction,
            )
            threshold_for_roi = self._representative_boundary_target(
                boundary_targets,
                target_boundary_idx=self.target_boundary_idx,
                boundary_weights=self.boundary_weights,
            )

        score = self._postprocess_pointwise_score(
            score,
            summary,
            Xt,
            X_in,
            name="qHeteroOrdinalBoundaryVarianceAcquisition",
            threshold_for_roi=threshold_for_roi,
        )
        out = self._maybe_reshape_output(self._reduce_q(score), original_batch_shape)
        self._check_output_shape(out, original_batch_shape, "qHeteroOrdinalBoundaryVarianceAcquisition")
        return out


class qHeteroOrdinalClassEntropyAcquisition(_BaseHeteroOrdinalLevelSetAcquisition):
    """heteroscedastic ordinal 用 class entropy acquisition。

    classification の qHeteroBinaryClassEntropyAcquisition に対応します。
    ordinal class probability の Shannon entropy を pointwise score として使います。
    """

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._prepare_eval()
        X_in = _ensure_q_batch(X)
        original_batch_shape = X_in.shape[:-2]
        Xt = self._apply_input_transform(X_in)
        summary = self._summary(X_in)

        probs = self._class_probs_from_summary_or_model(X_in, summary, Xt)
        score = _entropy_from_probs(probs, eps=self.eps)

        score = self._postprocess_pointwise_score(
            score,
            summary,
            Xt,
            X_in,
            name="qHeteroOrdinalClassEntropyAcquisition",
            threshold_for_roi=None,
        )
        out = self._maybe_reshape_output(self._reduce_q(score), original_batch_shape)
        self._check_output_shape(out, original_batch_shape, "qHeteroOrdinalClassEntropyAcquisition")
        return out


__all__ = [
    "HeteroOrdinalLevelSetScoreObjective",
    "BoundaryReduction",
    "qHeteroOrdinalLatentStraddleAcquisition",
    "qHeteroOrdinalICUAcquisition",
    "qHeteroOrdinalBoundaryVarianceAcquisition",
    "qHeteroOrdinalClassEntropyAcquisition",
]

