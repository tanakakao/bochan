from __future__ import annotations

import math
from typing import Callable, Literal, Optional

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.models.model import Model
from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor

from bochan.acquisition.binary.base import (
    LargeQStrategy,
    ReductionType,
    UncertaintyScoreType,
    _BinaryClassificationAcqBase,
)

try:
    from bochan.acquisition.binary.base import (
        ROIWeightMode,
        ROICombineType,
    )
except ImportError:  # fallback for older base.py
    ROIWeightMode = Literal["none", "prob_above", "prob_below", "prob_interval", "custom"]
    ROICombineType = Literal["multiply", "add"]

try:
    from ...objective.binary import BinaryClassificationScoreObjectiveMixin
except ImportError:  # fallback when installed under bochan.acquisition
    from bochan.acquisition.objective.binary import BinaryClassificationScoreObjectiveMixin

from ._utils import (
    _align_pointwise_score_to_X,
    _apply_objective_to_pointwise_score,
)


class _BALDAcquisition(BinaryClassificationScoreObjectiveMixin, _BinaryClassificationAcqBase):
    """
    BALD: H[E_w p(y|x,w)] - E_w[H[p(y|x,w)]].

    The objective is applied to the pointwise BALD score before _reduce_q.
    This supports InputPerturbation risk aggregation such as mean / VaR / CVaR.
    """

    def __init__(
        self,
        model,
        num_samples: int = 16,
        reduction: ReductionType = "mean",
        pending_penalty_weight: float = 0.0,
        pending_penalty_beta: float = 10.0,
        eps: float = 1e-6,
        roi_mode: ROIWeightMode = "none",
        roi_combine: ROICombineType = "multiply",
        roi_threshold: float = 0.5,
        roi_target_prob: float = 0.8,
        roi_interval: Optional[tuple[float, float]] = None,
        roi_beta: float = 20.0,
        roi_bandwidth: float = 0.15,
        roi_min_weight: float = 0.0,
        roi_weight_scale: float = 1.0,
        roi_aggregate_reduction: ReductionType = "mean",
        roi_weight_fn: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
        objective: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
    ):
        super().__init__(
            model=model,
            reduction=reduction,
            pending_penalty_weight=pending_penalty_weight,
            pending_penalty_beta=pending_penalty_beta,
            eps=eps,
            roi_mode=roi_mode,
            roi_combine=roi_combine,
            roi_threshold=roi_threshold,
            roi_target_prob=roi_target_prob,
            roi_interval=roi_interval,
            roi_beta=roi_beta,
            roi_bandwidth=roi_bandwidth,
            roi_min_weight=roi_min_weight,
            roi_weight_scale=roi_weight_scale,
            roi_aggregate_reduction=roi_aggregate_reduction,
            roi_weight_fn=roi_weight_fn,
        )
        self.num_samples = int(num_samples)
        self._set_classification_score_objective(objective)

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._prepare_eval()
        X_in = X if X.dim() > 2 else X.unsqueeze(0)
        original_batch_shape = X_in.shape[:-2]

        probs, _, Xt = self._pointwise_latent_probs(X, num_samples=self.num_samples)
        entropy_conditional = self._binary_entropy(probs, self.eps).mean(dim=0)
        mean_prob = probs.mean(dim=0)
        mean_entropy = self._binary_entropy(mean_prob, self.eps)
        score = mean_entropy - entropy_conditional

        score = self._apply_roi_weight_per_point(score, mean_prob, Xt)
        score = score - self._pending_penalty_per_point(Xt)
        score = _align_pointwise_score_to_X(
            score,
            Xt,
            name="BALD score before objective",
            reduce_extra="sum",
        )
        score = _apply_objective_to_pointwise_score(
            self,
            score,
            raw_X=X_in,
            expanded_X=Xt,
            name="BALD",
        )

        out = self._reduce_q(score)
        self._check_output_shape(out, original_batch_shape, "BALD")
        return out


class _JointQBALDAcquisitionBinary(BinaryClassificationScoreObjectiveMixin, _BinaryClassificationAcqBase):
    """
    Binary classification joint qBALD-like acquisition.

    The score is already aggregated over the q-batch. InputPerturbation risk
    aggregation should usually be handled by pointwise acquisition functions,
    not by applying a q*n_w -> q objective to this joint score.
    """

    def __init__(
        self,
        model,
        num_samples: int = 32,
        max_joint_q: int = 8,
        large_q_strategy: LargeQStrategy = "per_point",
        pending_penalty_weight: float = 0.0,
        pending_penalty_beta: float = 10.0,
        eps: float = 1e-6,
        roi_mode: ROIWeightMode = "none",
        roi_combine: ROICombineType = "multiply",
        roi_threshold: float = 0.5,
        roi_target_prob: float = 0.8,
        roi_interval: Optional[tuple[float, float]] = None,
        roi_beta: float = 20.0,
        roi_bandwidth: float = 0.15,
        roi_min_weight: float = 0.0,
        roi_weight_scale: float = 1.0,
        roi_aggregate_reduction: ReductionType = "mean",
        roi_weight_fn: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
        objective: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
    ):
        super().__init__(
            model=model,
            reduction="sum",
            pending_penalty_weight=pending_penalty_weight,
            pending_penalty_beta=pending_penalty_beta,
            eps=eps,
            roi_mode=roi_mode,
            roi_combine=roi_combine,
            roi_threshold=roi_threshold,
            roi_target_prob=roi_target_prob,
            roi_interval=roi_interval,
            roi_beta=roi_beta,
            roi_bandwidth=roi_bandwidth,
            roi_min_weight=roi_min_weight,
            roi_weight_scale=roi_weight_scale,
            roi_aggregate_reduction=roi_aggregate_reduction,
            roi_weight_fn=roi_weight_fn,
        )
        self.num_samples = int(num_samples)
        self.max_joint_q = int(max_joint_q)
        self.large_q_strategy = large_q_strategy
        self._set_classification_score_objective(objective)

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._prepare_eval()
        probs, batch_shape, _, Xt = self._joint_latent_probs(X, num_samples=self.num_samples)
        mean_prob = probs.mean(dim=0)

        joint_entropy = self._joint_predictive_entropy_binary(
            probs, max_joint_q=self.max_joint_q, large_q_strategy=self.large_q_strategy
        )
        cond_entropy = self._conditional_entropy_given_w(probs)
        out = joint_entropy - cond_entropy

        out = self._apply_roi_weight_aggregated(out, mean_prob, Xt)
        out = out - self._pending_penalty_aggregated(Xt, reduction="sum")
        # Do not apply q*n_w -> q risk objective here. This score is already
        # joint/aggregated, not pointwise over q*n_w.

        self._check_output_shape(out, batch_shape, "JointQBALD")
        return out


class _GreedyJointQBALDAcquisitionBinary(BinaryClassificationScoreObjectiveMixin, _BinaryClassificationAcqBase):
    """Binary classification greedy qBALD acquisition."""

    def __init__(
        self,
        model,
        num_samples: int = 32,
        max_joint_points: int = 8,
        large_q_strategy: LargeQStrategy = "per_point",
        eps: float = 1e-6,
        # --- ROI weighting ---
        roi_mode: ROIWeightMode = "none",
        roi_combine: ROICombineType = "multiply",
        roi_threshold: float = 0.5,
        roi_target_prob: float = 0.8,
        roi_interval: Optional[tuple[float, float]] = None,
        roi_beta: float = 20.0,
        roi_bandwidth: float = 0.15,
        roi_min_weight: float = 0.0,
        roi_weight_scale: float = 1.0,
        roi_aggregate_reduction: ReductionType = "mean",
        roi_weight_fn: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
        objective: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
    ):
        super().__init__(
            model=model,
            reduction="sum",
            pending_penalty_weight=0.0,
            pending_penalty_beta=10.0,
            eps=eps,
            roi_mode=roi_mode,
            roi_combine=roi_combine,
            roi_threshold=roi_threshold,
            roi_target_prob=roi_target_prob,
            roi_interval=roi_interval,
            roi_beta=roi_beta,
            roi_bandwidth=roi_bandwidth,
            roi_min_weight=roi_min_weight,
            roi_weight_scale=roi_weight_scale,
            roi_aggregate_reduction=roi_aggregate_reduction,
            roi_weight_fn=roi_weight_fn,
        )
        self.num_samples = int(num_samples)
        self.max_joint_points = int(max_joint_points)
        self.large_q_strategy = large_q_strategy
        self._set_classification_score_objective(objective)

    @staticmethod
    def _expand_pending_to_batch(X_pending: Tensor, batch_shape: torch.Size) -> Tensor:
        if X_pending.ndim == 2:
            m, d = X_pending.shape
            return X_pending.view(*([1] * len(batch_shape)), m, d).expand(*batch_shape, m, d)
        if X_pending.ndim >= 3:
            m, d = X_pending.shape[-2], X_pending.shape[-1]
            Xp = X_pending.reshape(*([1] * len(batch_shape)), m, d)
            return Xp.expand(*batch_shape, m, d)
        raise ValueError(f"Unexpected X_pending shape: {tuple(X_pending.shape)}")

    def _joint_bald_score(self, X: Tensor) -> Tensor:
        probs, batch_shape, _, _ = self._joint_latent_probs(X, num_samples=self.num_samples)
        joint_entropy = self._joint_predictive_entropy_binary(
            probs, max_joint_q=self.max_joint_points, large_q_strategy=self.large_q_strategy,
        )
        cond_entropy = self._conditional_entropy_given_w(probs)
        out = joint_entropy - cond_entropy
        self._check_output_shape(out, batch_shape, "GreedyJointQBALD")
        return out

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._prepare_eval()
        batch_shape = X.shape[:-2]
        Xp = getattr(self, "X_pending", None)
        if Xp is not None:
            Xp = Xp.detach()

        probs_x, _, Xt = self._pointwise_latent_probs(X, num_samples=self.num_samples)
        mean_prob_x = probs_x.mean(dim=0)

        if Xp is None or Xp.numel() == 0:
            out = self._joint_bald_score(X)
            out = self._apply_roi_weight_aggregated(out, mean_prob_x, Xt)
            # Do not apply q*n_w -> q risk objective to an already aggregated
            # greedy joint score.
            self._check_output_shape(out, batch_shape, "GreedyJointQBALD")
            return out

        Xp_batch = self._expand_pending_to_batch(Xp, batch_shape)
        score_pending = self._joint_bald_score(Xp_batch)
        X_all = torch.cat([Xp_batch, X], dim=-2)
        score_all = self._joint_bald_score(X_all)

        out = score_all - score_pending
        out = self._apply_roi_weight_aggregated(out, mean_prob_x, Xt)
        # Do not apply q*n_w -> q risk objective to an already aggregated
        # greedy joint score.
        self._check_output_shape(out, batch_shape, "GreedyJointQBALD")
        return out


class _UncertaintySamplingClassifierAcquisition(BinaryClassificationScoreObjectiveMixin, _BinaryClassificationAcqBase):
    """Binary classification uncertainty sampling acquisition."""

    def __init__(
        self,
        model,
        reduction: ReductionType = "mean",
        score_type: UncertaintyScoreType = "variance",
        pending_penalty_weight: float = 0.0,
        pending_penalty_beta: float = 10.0,
        apply_sigmoid_if_needed: bool = False,
        eps: float = 1e-6,
        objective: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
    ):
        super().__init__(
            model=model,
            reduction=reduction,
            pending_penalty_weight=pending_penalty_weight,
            pending_penalty_beta=pending_penalty_beta,
            eps=eps,
        )
        self.score_type = score_type
        self.apply_sigmoid_if_needed = bool(apply_sigmoid_if_needed)
        self._set_classification_score_objective(objective)

    def _normalize_prob_shape(self, p: Tensor, X: Tensor) -> Tensor:
        """
        Normalize posterior.mean to either (*batch, q) or (*batch, q * n_w).
        """
        X = self._ensure_q_batch(X)

        batch_shape = X.shape[:-2]
        q = X.shape[-2]

        expected = batch_shape + torch.Size([q])

        if p.shape == expected:
            return p

        # (..., q, 1) -> (..., q)
        if p.ndim == X.ndim and p.shape[-1] == 1:
            p_squeezed = p.squeeze(-1)
            if p_squeezed.shape == expected:
                return p_squeezed

        # InputPerturbation: p.shape = (*batch, q * n_w)
        if p.ndim == len(batch_shape) + 1 and p.shape[:-1] == batch_shape:
            q_expanded = p.shape[-1]
            if q_expanded >= q and q_expanded % q == 0:
                return p

        # (..., q * n_w, 1) -> (..., q * n_w)
        if p.ndim == len(batch_shape) + 2 and p.shape[:-2] == batch_shape and p.shape[-1] == 1:
            p_squeezed = p.squeeze(-1)
            q_expanded = p_squeezed.shape[-1]
            if q_expanded >= q and q_expanded % q == 0:
                return p_squeezed

        if p.numel() == math.prod(expected):
            return p.reshape(*expected)

        raise RuntimeError(
            f"Unsupported posterior mean shape for binary classification: "
            f"X.shape={tuple(X.shape)}, posterior.mean.shape={tuple(p.shape)}"
        )

    def _to_probability(self, p: Tensor) -> Tensor:
        pmin = p.min().item()
        pmax = p.max().item()
        if 0.0 <= pmin and pmax <= 1.0:
            return p.clamp(self.eps, 1.0 - self.eps)
        if self.apply_sigmoid_if_needed:
            return torch.sigmoid(p).clamp(self.eps, 1.0 - self.eps)
        raise RuntimeError(
            f"posterior.mean is not in [0,1] (min={pmin:.4g}, max={pmax:.4g}). "
            "This acquisition assumes probability output. "
            "Either fix the classifier wrapper or set apply_sigmoid_if_needed=True."
        )

    def _uncertainty_score(self, p: Tensor) -> Tensor:
        if self.score_type == "variance":
            return p * (1.0 - p)
        if self.score_type == "entropy":
            return self._binary_entropy(p, self.eps)
        if self.score_type == "least_confidence":
            return 1.0 - torch.maximum(p, 1.0 - p)
        raise ValueError(f"Unknown score_type: {self.score_type}")

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._prepare_eval()

        X = self._ensure_q_batch(X)
        original_batch_shape = X.shape[:-2]

        prob_fn = getattr(self.model, "probability_posterior", None)
        posterior = prob_fn(X) if callable(prob_fn) else self.model.posterior(X)
        p = self._normalize_prob_shape(posterior.mean, X)
        p = self._to_probability(p)

        score = self._uncertainty_score(p)
        Xt = self._apply_input_transform(X)

        score = _align_pointwise_score_to_X(
            score,
            Xt,
            name="UncertaintySampling score before penalty",
            reduce_extra="sum",
        )

        penalty = self._pending_penalty_per_point(Xt)
        if penalty.shape == score.shape:
            score = score - penalty

        score = _align_pointwise_score_to_X(
            score,
            Xt,
            name="UncertaintySampling score before objective",
            reduce_extra="sum",
        )
        score = _apply_objective_to_pointwise_score(
            self,
            score,
            raw_X=X,
            expanded_X=Xt,
            name="UncertaintySampling",
        )

        out = self._reduce_q(score)
        self._check_output_shape(out, original_batch_shape, "UncertaintySampling")
        return out



# =========================================================
# Unified active-learning family names
# =========================================================
class _qBinaryPredictiveEntropyAcquisition(_UncertaintySamplingClassifierAcquisition):
    """Binary classification predictive entropy acquisition."""

    def __init__(self, model, **kwargs):
        kwargs.pop("score_type", None)
        super().__init__(model=model, score_type="entropy", **kwargs)


class _qBinaryProbabilityVarianceAcquisition(_UncertaintySamplingClassifierAcquisition):
    """Binary classification probability variance p(1-p) acquisition."""

    def __init__(self, model, **kwargs):
        kwargs.pop("score_type", None)
        super().__init__(model=model, score_type="variance", **kwargs)


class _qBinaryMarginUncertaintyAcquisition(_UncertaintySamplingClassifierAcquisition):
    """Binary classification least-confidence / margin uncertainty acquisition."""

    def __init__(self, model, **kwargs):
        kwargs.pop("score_type", None)
        super().__init__(model=model, score_type="least_confidence", **kwargs)



# Canonical unified names






class qBinaryPredictiveEntropy(_qBinaryPredictiveEntropyAcquisition):
    """classification 用 predictive entropy acquisition。予測分布の曖昧さが大きい点を選びます。
    
    Forward Args:
        X: 候補点。shape は通常 `batch_shape x q x d` です。
    
    Returns:
        Tensor: `batch_shape` の acquisition value。`optimize_acqf` はこの値を最大化します。
    
    Notes:
        予測が曖昧な点を探索したい場合の基本的な active learning acquisition です。
    """
    pass


class qBinaryBALD(_BALDAcquisition):
    """classification 用 BALD / mutual-information acquisition。モデル不確実性を減らす情報量の大きい点を選びます。
    
    Forward Args:
        X: 候補点。shape は通常 `batch_shape x q x d` です。
    
    Returns:
        Tensor: `batch_shape` の acquisition value。`optimize_acqf` はこの値を最大化します。
    
    Notes:
        BALD は predictive entropy から条件付き entropy を引いた情報利得として解釈できます。
    """
    pass


class qBinaryJointBALD(_JointQBALDAcquisitionBinary):
    """classification 用 BALD / mutual-information acquisition。モデル不確実性を減らす情報量の大きい点を選びます。
    
    Forward Args:
        X: 候補点。shape は通常 `batch_shape x q x d` です。
    
    Returns:
        Tensor: `batch_shape` の acquisition value。`optimize_acqf` はこの値を最大化します。
    
    Notes:
        BALD は predictive entropy から条件付き entropy を引いた情報利得として解釈できます。
    """
    pass


class qBinaryGreedyJointBALD(_GreedyJointQBALDAcquisitionBinary):
    """classification 用 BALD / mutual-information acquisition。モデル不確実性を減らす情報量の大きい点を選びます。
    
    Forward Args:
        X: 候補点。shape は通常 `batch_shape x q x d` です。
    
    Returns:
        Tensor: `batch_shape` の acquisition value。`optimize_acqf` はこの値を最大化します。
    
    Notes:
        BALD は predictive entropy から条件付き entropy を引いた情報利得として解釈できます。
    """
    pass


class qBinaryProbabilityVariance(_qBinaryProbabilityVarianceAcquisition):
    """classification 用の軽量 probability variance acquisition。

    候補点上の `p(1-p)` を使う proxy であり、mc_points は不要です。
    
    Forward Args:
        X: 候補点。shape は通常 `batch_shape x q x d` です。
    
    Returns:
        Tensor: `batch_shape` の acquisition value。`optimize_acqf` はこの値を最大化します。
    """
    pass


class qBinaryMarginUncertainty(_qBinaryMarginUncertaintyAcquisition):
    """classification 用 margin uncertainty acquisition。決定境界または class 境界に近い点を選びます。
    
    Forward Args:
        X: 候補点。shape は通常 `batch_shape x q x d` です。
    
    Returns:
        Tensor: `batch_shape` の acquisition value。`optimize_acqf` はこの値を最大化します。
    """
    pass



# =========================================================
# Explicit IPV family
# =========================================================
def _ensure_q_batch_for_ipv(X: Tensor) -> Tensor:
    """X を pending / fantasy NIPV 用に (..., q, d) へ揃える。"""
    if not torch.is_tensor(X):
        raise TypeError(f"X must be Tensor. Got {type(X)}.")
    if X.ndim == 1:
        return X.view(1, 1, -1)
    if X.ndim == 2:
        return X.unsqueeze(0)
    return X


def _coerce_reference_to_tensor_for_ipv(
    X_ref,
    *,
    ref: Optional[Tensor] = None,
) -> Optional[Tensor]:
    """X_pending / X_observed を Tensor または None に正規化する。"""
    if X_ref is None:
        return None

    if torch.is_tensor(X_ref):
        out = X_ref
    elif isinstance(X_ref, (list, tuple)):
        tensors = []
        for item in X_ref:
            if item is None:
                continue
            t = _coerce_reference_to_tensor_for_ipv(item, ref=ref)
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
            "X_pending / X_observed must be None, Tensor, list, or tuple. "
            f"Got {type(X_ref)}."
        )

    if ref is not None:
        out = out.to(device=ref.device, dtype=ref.dtype)

    # pending / observed / mc reference points are constants during acquisition optimization.
    return out.detach()


def _apply_input_transform_for_ipv(model: Model, X: Tensor) -> Tensor:
    """candidate / pending / observed を同じ距離計算空間へ写す。"""
    X = _ensure_q_batch_for_ipv(X)

    input_transform = getattr(model, "input_transform", None)
    if input_transform is not None:
        Xt = input_transform(X)
        if isinstance(Xt, tuple):
            Xt = Xt[0]
        return _ensure_q_batch_for_ipv(Xt)

    models = getattr(model, "models", None)
    if models is not None and len(models) > 0:
        input_transform = getattr(models[0], "input_transform", None)
        if input_transform is not None:
            Xt = input_transform(X)
            if isinstance(Xt, tuple):
                Xt = Xt[0]
            return _ensure_q_batch_for_ipv(Xt)

    return X


def _transform_reference_like_candidate_for_ipv(
    model: Model,
    X_ref,
    *,
    ref: Tensor,
) -> Optional[Tensor]:
    Xr = _coerce_reference_to_tensor_for_ipv(X_ref, ref=ref)
    if Xr is None or Xr.numel() == 0:
        return None
    Xr_t = _apply_input_transform_for_ipv(model, Xr)
    return Xr_t.to(device=ref.device, dtype=ref.dtype)


def _broadcast_reference_to_batch_for_ipv(X_ref: Tensor, batch_shape: torch.Size) -> Tensor:
    X_ref = _ensure_q_batch_for_ipv(X_ref)

    if X_ref.shape[:-2] == batch_shape:
        return X_ref

    try:
        return X_ref.expand(*batch_shape, X_ref.shape[-2], X_ref.shape[-1])
    except RuntimeError:
        X2d = X_ref.reshape(-1, X_ref.shape[-1])
        return X2d.view(*([1] * len(batch_shape)), X2d.shape[-2], X2d.shape[-1]).expand(
            *batch_shape,
            X2d.shape[-2],
            X2d.shape[-1],
        )


def _rbf_reference_penalty_aggregated_for_ipv(
    X: Tensor,
    X_ref: Optional[Tensor],
    beta: float,
) -> Tensor:
    """batch-level の reference penalty を返す。shape = batch_shape。"""
    X = _ensure_q_batch_for_ipv(X)

    if X_ref is None or X_ref.numel() == 0:
        return X.new_zeros(X.shape[:-2])

    X_ref = _broadcast_reference_to_batch_for_ipv(
        X_ref.to(device=X.device, dtype=X.dtype),
        X.shape[:-2],
    )

    d2 = torch.cdist(X, X_ref).pow(2)
    per_point = torch.exp(-float(beta) * d2).amax(dim=-1)
    return per_point.sum(dim=-1)


def _resolve_observed_X_for_ipv(
    model: Model,
    X_observed: Optional[Tensor] = None,
) -> Optional[Tensor]:
    if X_observed is not None:
        return X_observed

    for attr in ("train_inputs_raw", "train_X_original", "train_X"):
        x = getattr(model, attr, None)
        if x is not None:
            return x

    train_inputs = getattr(model, "train_inputs", None)
    if isinstance(train_inputs, tuple) and len(train_inputs) > 0:
        return train_inputs[0]

    return None


def _binary_values_to_probability_for_ipv(
    values: Tensor,
    *,
    apply_sigmoid_if_needed: bool,
    eps: float,
    name: str,
) -> Tensor:
    """posterior mean / samples を binary probability に変換する。"""
    if values.ndim > 0 and values.shape[-1] == 1:
        values = values.squeeze(-1)

    vmin = values.detach().min().item()
    vmax = values.detach().max().item()

    if 0.0 <= vmin and vmax <= 1.0:
        return values.clamp(eps, 1.0 - eps)

    if apply_sigmoid_if_needed:
        return torch.sigmoid(values).clamp(eps, 1.0 - eps)

    raise RuntimeError(
        f"{name} is not in [0, 1] (min={vmin:.4g}, max={vmax:.4g}). "
        "Set apply_sigmoid_if_needed=True or make the model posterior return probabilities."
    )


class qBinaryFantasyNegIntegratedPosteriorVariance(AcquisitionFunction):
    """
    binary classification 用の fantasy negative integrated posterior variance。

    注意:
        fantasy model を作り直すため、通常の L-BFGS-B などの勾配最適化には
        向きません。`optimize_func="evo"` などの勾配不要 optimizer での利用を
        想定します。

    `qBinaryProbabilityVariance` は候補点上の軽量 proxy です。
    この class は候補点で fantasy label を生成して `condition_on_observations` した後、
    `mc_points` 上の binary probability variance `p(1-p)` を積分する高コスト版です。

    Args:
        model: BoTorch 互換 model。`posterior` と `condition_on_observations` を想定。
        mc_points: 積分点集合。shape = n_mc x d。
        num_fantasies: fantasy label 数。
        conditioning_steps: `condition_on_observations(..., refit=True)` に渡す再学習 step 数。
        conditioning_lr: 再学習時の learning rate。
        conditioning_batch_size: 再学習時の batch size。
        apply_sigmoid_if_needed: posterior mean が latent 値の場合に sigmoid で確率化するか。
        pending_penalty_weight: X_pending 近傍を避ける penalty の強さ。
        pending_penalty_beta: pending penalty の距離減衰率。
        observed_penalty_weight: X_observed 近傍を避ける penalty の強さ。
        observed_penalty_beta: observed penalty の距離減衰率。
        X_pending: 評価中候補。
        X_observed: 観測済み点。
        eps: 数値安定化用の微小値。
    """

    def __init__(
        self,
        model: Model,
        mc_points: Tensor,
        num_fantasies: int = 8,
        conditioning_steps: int = 10,
        conditioning_lr: float | None = None,
        conditioning_batch_size: int | None = None,
        apply_sigmoid_if_needed: bool = True,
        pending_penalty_weight: float = 0.0,
        pending_penalty_beta: float = 10.0,
        observed_penalty_weight: float = 0.0,
        observed_penalty_beta: float = 10.0,
        X_pending: Optional[Tensor] = None,
        X_observed: Optional[Tensor] = None,
        eps: float = 1e-6,
    ) -> None:
        super().__init__(model=model)

        if mc_points.ndim != 2:
            raise ValueError(
                f"mc_points must be [N_mc, d], got shape={tuple(mc_points.shape)}."
            )

        ref_X = getattr(model, "train_X", None)
        if ref_X is None:
            train_inputs = getattr(model, "train_inputs", None)
            if isinstance(train_inputs, tuple) and len(train_inputs) > 0:
                ref_X = train_inputs[0]

        if ref_X is not None:
            mc_points = mc_points.to(device=ref_X.device, dtype=ref_X.dtype)

        self.register_buffer("mc_points", mc_points)
        self.num_fantasies = int(num_fantasies)
        self.conditioning_steps = int(conditioning_steps)
        self.conditioning_lr = conditioning_lr
        self.conditioning_batch_size = conditioning_batch_size
        self.apply_sigmoid_if_needed = bool(apply_sigmoid_if_needed)
        self.pending_penalty_weight = float(pending_penalty_weight)
        self.pending_penalty_beta = float(pending_penalty_beta)
        self.observed_penalty_weight = float(observed_penalty_weight)
        self.observed_penalty_beta = float(observed_penalty_beta)
        self.eps = float(eps)

        self.X_pending: Optional[Tensor] = None
        self.set_X_pending(X_pending)

        self.X_observed: Optional[Tensor] = None
        self.set_X_observed(X_observed)

    def set_X_pending(self, X_pending: Optional[Tensor] = None) -> None:
        self.X_pending = _coerce_reference_to_tensor_for_ipv(X_pending)

    def set_X_observed(self, X_observed: Optional[Tensor] = None) -> None:
        self.X_observed = _coerce_reference_to_tensor_for_ipv(
            _resolve_observed_X_for_ipv(self.model, X_observed)
        )

    def _prepare_eval(self) -> None:
        self.model.eval()
        likelihood = getattr(self.model, "likelihood", None)
        if hasattr(likelihood, "eval"):
            likelihood.eval()

    @torch.no_grad()
    def _sample_fantasy_labels(self, X: Tensor) -> Tensor:
        prob_fn = getattr(self.model, "probability_posterior", None)
        posterior = prob_fn(X) if callable(prob_fn) else self.model.posterior(X)
        prob = _binary_values_to_probability_for_ipv(
            posterior.mean,
            apply_sigmoid_if_needed=self.apply_sigmoid_if_needed,
            eps=self.eps,
            name="binary posterior mean",
        )

        # prob: q or batch_shape x q. ここでは X は通常 q x d。
        prob = prob.reshape(*prob.shape[:-1], prob.shape[-1]) if prob.ndim > 1 else prob
        fantasy_Y = []
        for _ in range(self.num_fantasies):
            yi = torch.bernoulli(prob).to(dtype=X.dtype)
            if yi.ndim == 1:
                yi = yi.unsqueeze(-1)
            fantasy_Y.append(yi)

        return torch.stack(fantasy_Y, dim=0)

    @torch.no_grad()
    def _integrated_probability_variance(self, fantasy_model: Model) -> Tensor:
        prob_fn = getattr(fantasy_model, "probability_posterior", None)
        posterior = prob_fn(self.mc_points) if callable(prob_fn) else fantasy_model.posterior(self.mc_points)
        prob = _binary_values_to_probability_for_ipv(
            posterior.mean,
            apply_sigmoid_if_needed=self.apply_sigmoid_if_needed,
            eps=self.eps,
            name="fantasy posterior mean",
        )
        return (prob * (1.0 - prob)).mean()

    def _aggregated_reference_penalty(self, X: Tensor) -> Tensor:
        Xt = _apply_input_transform_for_ipv(self.model, X)
        penalty = torch.zeros(Xt.shape[:-2], device=Xt.device, dtype=Xt.dtype)

        if self.pending_penalty_weight > 0.0:
            Xp_t = _transform_reference_like_candidate_for_ipv(self.model, self.X_pending, ref=Xt)
            penalty = penalty + self.pending_penalty_weight * _rbf_reference_penalty_aggregated_for_ipv(
                X=Xt,
                X_ref=Xp_t,
                beta=self.pending_penalty_beta,
            )

        if self.observed_penalty_weight > 0.0:
            Xobs_t = _transform_reference_like_candidate_for_ipv(self.model, self.X_observed, ref=Xt)
            penalty = penalty + self.observed_penalty_weight * _rbf_reference_penalty_aggregated_for_ipv(
                X=Xt,
                X_ref=Xobs_t,
                beta=self.observed_penalty_beta,
            )

        return penalty

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._prepare_eval()
        X = _ensure_q_batch_for_ipv(X)
        batch_shape = X.shape[:-2]
        X_flat = X.reshape(-1, X.shape[-2], X.shape[-1])

        out = []
        for Xb in X_flat:
            Xb = Xb.detach()
            fantasy_Y = self._sample_fantasy_labels(Xb)

            vals = []
            for f in range(self.num_fantasies):
                with torch.enable_grad():
                    fantasy_model = self.model.condition_on_observations(
                        X=Xb,
                        Y=fantasy_Y[f].detach(),
                        refit=True,
                        num_steps=self.conditioning_steps,
                        lr=self.conditioning_lr,
                        batch_size=self.conditioning_batch_size,
                        verbose=False,
                    )

                with torch.no_grad():
                    ipv = self._integrated_probability_variance(fantasy_model)
                vals.append(-ipv)

            out.append(torch.stack(vals).mean())

        score = torch.stack(out).reshape(*batch_shape)
        penalty = self._aggregated_reference_penalty(X)

        if penalty.shape != score.shape:
            if penalty.numel() == score.numel():
                penalty = penalty.reshape_as(score)
            else:
                raise RuntimeError(
                    "qBinaryFantasyNegIntegratedPosteriorVariance penalty shape mismatch. "
                    f"score.shape={tuple(score.shape)}, penalty.shape={tuple(penalty.shape)}"
                )

        return score - penalty


__all__ = [
    "qBinaryPredictiveEntropy",
    "qBinaryBALD",
    "qBinaryJointBALD",
    "qBinaryGreedyJointBALD",
    "qBinaryProbabilityVariance",
    "qBinaryMarginUncertainty",
    "qBinaryFantasyNegIntegratedPosteriorVariance",
]
