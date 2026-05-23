from __future__ import annotations

import math
from typing import Callable, Literal, Optional

import torch
from torch import Tensor
from torch.distributions import Categorical

from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.models.model import Model
from botorch.sampling.base import MCSampler
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.transforms import average_over_ensemble_models, t_batch_mode_transform

from bochan.likelihoods.ordinal import OrdinalLogitLikelihood


RiskType = Optional[Literal["var", "cvar"]]
ReductionType = Literal["mean", "sum"]


# =========================================================
# Likelihood / objective utilities
# =========================================================
def _is_ordinal_likelihood(obj) -> bool:
    return (
        obj is not None
        and hasattr(obj, "marginal_class_probs")
        and hasattr(obj, "class_probs_from_f")
    )


def _resolve_ordinal_likelihood(
    model: Model,
    ordinal_likelihood: Optional[OrdinalLogitLikelihood] = None,
) -> OrdinalLogitLikelihood:
    """
    acquisition で使う ordinal likelihood を解決する。

    優先順位:
        1. 明示引数 ordinal_likelihood
        2. model.ordinal_likelihood
        3. model.likelihood
        4. model.latent_model / base_model / model の ordinal_likelihood / likelihood
        5. model.models[*] の ordinal_likelihood / likelihood

    Notes:
        Multi-output ordinal model で複数の ordinal likelihood が見つかる場合は、
        自動選択せずに明示指定を要求する。
    """
    if ordinal_likelihood is not None:
        return ordinal_likelihood

    candidates = [
        getattr(model, "ordinal_likelihood", None),
        getattr(model, "likelihood", None),
    ]

    for attr in ("latent_model", "base_model", "model"):
        inner = getattr(model, attr, None)
        if inner is not None:
            candidates.append(getattr(inner, "ordinal_likelihood", None))
            candidates.append(getattr(inner, "likelihood", None))

    for cand in candidates:
        if _is_ordinal_likelihood(cand):
            return cand

    models = getattr(model, "models", None)
    if models is not None:
        likelihoods = []
        for m in models:
            for cand in (
                getattr(m, "ordinal_likelihood", None),
                getattr(m, "likelihood", None),
            ):
                if _is_ordinal_likelihood(cand):
                    likelihoods.append(cand)

        if len(likelihoods) == 1:
            return likelihoods[0]

        if len(likelihoods) > 1:
            raise ValueError(
                "Multiple ordinal likelihoods were found in model.models. "
                "Pass ordinal_likelihood explicitly or use a dedicated "
                "multi-output ordinal acquisition."
            )

    raise ValueError(
        "ordinal_likelihood was not provided and could not be found from model. "
        "Expected model.ordinal_likelihood or model.likelihood to implement "
        "`marginal_class_probs` and `class_probs_from_f`."
    )


class OrdinalScoreObjective(torch.nn.Module):
    """
    ordinal active-learning acquisition の pointwise score に作用する objective。

    posterior samples ではなく、entropy / BALD / utility variance などで
    計算済みの acquisition score に作用する。

    InputPerturbation では:
        score: (*batch, q * n_w) -> (*batch, q)

    Args:
        n_w: 1 candidate あたりの input perturbation replica 数。
        risk_type: None / "var" / "cvar"。
        alpha: VaR / CVaR の tail 比率。
        maximize: acquisition を最大化する前提なら True。
        weight: score の重み。
        sign: score の符号。通常は 1.0。
    """

    def __init__(
        self,
        n_w: Optional[int] = None,
        risk_type: RiskType = None,
        alpha: float = 0.5,
        maximize: bool = True,
        weight: float = 1.0,
        sign: float = 1.0,
    ) -> None:
        super().__init__()
        self.n_w = None if n_w is None else int(n_w)
        self.risk_type = risk_type
        self.alpha = float(alpha)
        self.maximize = bool(maximize)
        self.weight = float(weight)
        self.sign = float(sign)

        if self.risk_type not in (None, "var", "cvar"):
            raise ValueError(f"Unknown risk_type: {self.risk_type}")
        if self.risk_type is not None and self.n_w is None:
            raise ValueError("risk_type is specified, but n_w is None.")
        if self.risk_type is not None and not (0.0 < self.alpha <= 1.0):
            raise ValueError("alpha must be in (0, 1].")

    def forward(self, score: Tensor, X: Optional[Tensor] = None) -> Tensor:
        score = score * self.sign * self.weight

        if self.n_w is None or self.n_w <= 1:
            return score

        # aggregated scalar score を n_w 方向と誤認しないための guard。
        if X is not None:
            X_in = X if X.ndim > 2 else X.unsqueeze(0)
            if tuple(score.shape) == tuple(X_in.shape[:-2]):
                return score

        q_expanded = score.shape[-1]
        if q_expanded % self.n_w != 0:
            raise RuntimeError(
                f"score.shape[-1] must be divisible by n_w. "
                f"Got score.shape={tuple(score.shape)}, n_w={self.n_w}."
            )

        q = q_expanded // self.n_w
        score_w = score.reshape(*score.shape[:-1], q, self.n_w)

        if self.risk_type is None:
            return score_w.mean(dim=-1)

        # maximize=True の worst tail は小さい score 側。
        descending = not self.maximize
        sorted_score = torch.sort(score_w, dim=-1, descending=descending).values
        k = max(1, int(math.ceil(self.n_w * self.alpha)))
        tail = sorted_score[..., :k]

        if self.risk_type == "var":
            return tail[..., -1]
        if self.risk_type == "cvar":
            return tail.mean(dim=-1)

        raise ValueError(f"Unknown risk_type: {self.risk_type}")


_OrdinalScoreObjective = OrdinalScoreObjective


def _apply_ordinal_objective_to_pointwise_score(
    owner,
    score: Tensor,
    raw_X: Optional[Tensor] = None,
    name: str = "OrdinalAcquisition",
) -> Tensor:
    objective = getattr(owner, "objective", None)
    if objective is None:
        return score

    try:
        out = objective(score, X=raw_X)
    except TypeError:
        out = objective(score)

    if not torch.is_tensor(out):
        raise RuntimeError(f"{name}: objective must return a Tensor. Got {type(out)}.")
    return out


def ordinal_entropy_from_probs(probs: Tensor, eps: float = 1e-12) -> Tensor:
    probs = probs.clamp_min(eps)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(eps)
    return -(probs * probs.log()).sum(dim=-1)


# =========================================================
# Shape utilities
# =========================================================
def _ensure_q_batch(X: Tensor) -> Tensor:
    if not torch.is_tensor(X):
        raise TypeError(f"X must be Tensor. Got {type(X)}.")
    if X.ndim == 1:
        return X.view(1, 1, -1)
    if X.ndim == 2:
        return X.unsqueeze(0)
    return X


def _reduce_extra_batch_dims(
    tensor: Tensor,
    X: Tensor,
    n_trailing_keep: int,
    *,
    reduce_extra: Literal["mean", "sum"] = "mean",
) -> Tensor:
    """
    tensor の余分な ensemble / MC / fully Bayesian batch 次元を落とす。

    Args:
        tensor: score / probability tensor。
        X: 参照 candidate tensor。shape は (..., q_like, d)。
        n_trailing_keep: tensor の末尾で保持する次元数。
        reduce_extra: 余分な次元の集約方法。
    """
    out = tensor
    X = _ensure_q_batch(X)
    x_batch_shape = tuple(X.shape[:-2])
    target_ndim = len(x_batch_shape) + n_trailing_keep

    while out.ndim > target_ndim:
        prefix = tuple(out.shape[:-n_trailing_keep]) if n_trailing_keep > 0 else tuple(out.shape)

        if len(x_batch_shape) == 0:
            reduce_dim = 0
        else:
            match_start = None
            max_start = len(prefix) - len(x_batch_shape)
            for s in range(max_start + 1):
                if tuple(prefix[s : s + len(x_batch_shape)]) == x_batch_shape:
                    match_start = s
                    break

            if match_start is None:
                reduce_dim = max(out.ndim - n_trailing_keep - 1, 0)
            else:
                protected = set(range(match_start, match_start + len(x_batch_shape)))
                extra_dims = [i for i in range(len(prefix)) if i not in protected]
                if not extra_dims:
                    break
                reduce_dim = extra_dims[0]

        if reduce_extra == "sum":
            out = out.sum(dim=reduce_dim)
        elif reduce_extra == "mean":
            out = out.mean(dim=reduce_dim)
        else:
            raise ValueError(f"Unknown reduce_extra: {reduce_extra}")

    return out


def _reduce_probs_to_match_X(
    probs: Tensor,
    X: Tensor,
    eps: float = 1e-12,
) -> Tensor:
    """
    probs を (..., q_like, C) に落とす。
    InputPerturbation ありでは q_like = q * n_w になり得る。
    """
    out = _reduce_extra_batch_dims(probs, X=X, n_trailing_keep=2, reduce_extra="mean")
    out = out.clamp_min(eps)
    return out / out.sum(dim=-1, keepdim=True).clamp_min(eps)


def _reduce_pointwise_scores_to_match_X(
    score_like: Tensor,
    X: Tensor,
    *,
    reduce_extra: Literal["mean", "sum"] = "mean",
) -> Tensor:
    """
    score_like を (..., q_like) に落とす。
    InputPerturbation ありでは q_like = q * n_w になり得る。
    """
    return _reduce_extra_batch_dims(
        score_like,
        X=X,
        n_trailing_keep=1,
        reduce_extra=reduce_extra,
    )


def _align_pointwise_score_to_X(
    score: Tensor,
    X_like: Tensor,
    *,
    name: str,
    reduce_extra: Literal["mean", "sum"] = "mean",
) -> Tensor:
    """
    pointwise score を X_like.shape[:-1] に揃える。

    Args:
        score: pointwise acquisition score。
        X_like: input_transform 後の candidate。shape = (..., q_like, d)。
        name: error message 用。
        reduce_extra: 余分な batch / MC 次元の集約方法。

    Returns:
        Tensor: shape = X_like.shape[:-1]
    """
    X_like = _ensure_q_batch(X_like)
    expected = X_like.shape[:-1]

    if score.shape == expected:
        return score

    # (..., q_like, 1) -> (..., q_like)
    if score.ndim == len(expected) + 1 and score.shape[-1] == 1:
        squeezed = score.squeeze(-1)
        if squeezed.shape == expected:
            return squeezed

    score = _reduce_pointwise_scores_to_match_X(
        score,
        X_like,
        reduce_extra=reduce_extra,
    )

    if score.shape == expected:
        return score

    if score.numel() == math.prod(expected):
        return score.reshape(*expected)

    raise RuntimeError(
        f"{name}: failed to align pointwise score to X. "
        f"score.shape={tuple(score.shape)}, X_like.shape={tuple(X_like.shape)}, "
        f"expected_score_shape={tuple(expected)}."
    )


# =========================================================
# Pending / observed penalty utilities
# =========================================================
def _coerce_reference_to_tensor(
    X_ref,
    *,
    ref: Optional[Tensor] = None,
) -> Optional[Tensor]:
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
            "X_pending / X_observed must be None, Tensor, list, or tuple. "
            f"Got {type(X_ref)}."
        )

    if ref is not None:
        out = out.to(device=ref.device, dtype=ref.dtype)
    return out


_coerce_pending_to_tensor = _coerce_reference_to_tensor


def _apply_input_transform_for_reference(model: Model, X: Tensor) -> Tensor:
    """
    candidate / pending / observed を同じ距離計算空間へ写す。

    Notes:
        - input_transform が候補を q * n_w に展開する場合、
          penalty も同じ q_like 空間で計算される。
        - ModelList 系では先頭 model の input_transform を参照する。
    """
    X = _ensure_q_batch(X)

    it = getattr(model, "input_transform", None)
    if it is not None:
        Xt = it(X)
        if isinstance(Xt, tuple):
            Xt = Xt[0]
        return _ensure_q_batch(Xt)

    models = getattr(model, "models", None)
    if models is not None and len(models) > 0:
        it = getattr(models[0], "input_transform", None)
        if it is not None:
            Xt = it(X)
            if isinstance(Xt, tuple):
                Xt = Xt[0]
            return _ensure_q_batch(Xt)

    return X


_apply_input_transform_for_pending = _apply_input_transform_for_reference


def _transform_reference_like_candidate(
    model: Model,
    X_ref,
    *,
    ref: Tensor,
) -> Optional[Tensor]:
    Xr = _coerce_reference_to_tensor(X_ref, ref=ref)
    if Xr is None or Xr.numel() == 0:
        return None

    Xr_t = _apply_input_transform_for_reference(model, Xr)
    return Xr_t.to(device=ref.device, dtype=ref.dtype)


_transform_pending_like_candidate = _transform_reference_like_candidate


def _broadcast_reference_to_batch(X_ref: Tensor, batch_shape: torch.Size) -> Tensor:
    X_ref = _ensure_q_batch(X_ref)

    if X_ref.shape[:-2] == batch_shape:
        return X_ref

    try:
        return X_ref.expand(*batch_shape, X_ref.shape[-2], X_ref.shape[-1])
    except RuntimeError:
        # 参照点側の batch 次元が candidate と合わない場合は flatten して unbatched 扱いにする。
        X2d = X_ref.reshape(-1, X_ref.shape[-1])
        return X2d.view(*([1] * len(batch_shape)), X2d.shape[-2], X2d.shape[-1]).expand(
            *batch_shape,
            X2d.shape[-2],
            X2d.shape[-1],
        )


def _rbf_reference_penalty_per_point(
    X: Tensor,
    X_ref: Optional[Tensor],
    beta: float,
) -> Tensor:
    """
    各 candidate 点ごとの RBF reference penalty を返す。

    Args:
        X: transformed candidate. shape = batch_shape x q_like x d
        X_ref: transformed reference. shape = m x d or batch_shape x m x d
        beta: exp(-beta * distance^2) の beta。

    Returns:
        Tensor: shape = batch_shape x q_like
    """
    X = _ensure_q_batch(X)
    if X_ref is None or X_ref.numel() == 0:
        return X.new_zeros(X.shape[:-1])

    X_ref = _broadcast_reference_to_batch(
        X_ref.to(device=X.device, dtype=X.dtype),
        X.shape[:-2],
    )

    d2 = torch.cdist(X, X_ref).pow(2)
    return torch.exp(-float(beta) * d2).amax(dim=-1)


def _rbf_reference_penalty_aggregated(
    X: Tensor,
    X_ref: Optional[Tensor],
    beta: float,
    reduction: ReductionType = "sum",
) -> Tensor:
    per_point = _rbf_reference_penalty_per_point(X=X, X_ref=X_ref, beta=beta)
    if reduction == "mean":
        return per_point.mean(dim=-1)
    if reduction == "sum":
        return per_point.sum(dim=-1)
    raise ValueError(f"Unknown reduction: {reduction}")


_rbf_pending_penalty = _rbf_reference_penalty_aggregated


def _resolve_observed_X(
    model: Model,
    X_observed: Optional[Tensor] = None,
) -> Optional[Tensor]:
    if X_observed is not None:
        return X_observed

    for attr in ("train_X_original", "train_X", "train_inputs_raw"):
        x = getattr(model, attr, None)
        if x is not None:
            return x

    x = getattr(model, "train_inputs", None)
    if isinstance(x, tuple) and len(x) > 0:
        return x[0]

    return None


# =========================================================
# Base class
# =========================================================
class _qOrdinalActiveLearningBase(MCAcquisitionFunction):
    """
    ordinal active-learning acquisition の共通 base。

    pointwise 系 acquisition の標準順序:
        1. posterior / likelihood から pointwise score を計算
        2. input_transform 後の空間に合わせて score shape を整列
        3. pending / observed penalty を pointwise に差し引く
        4. objective を pointwise score に適用
        5. q 方向を reduction で集約
    """

    def __init__(
        self,
        model: Model,
        ordinal_likelihood: Optional[OrdinalLogitLikelihood] = None,
        sampler: Optional[MCSampler] = None,
        reduction: ReductionType = "mean",
        pending_penalty_weight: float = 0.0,
        pending_penalty_beta: float = 10.0,
        observed_penalty_weight: float = 0.0,
        observed_penalty_beta: float = 10.0,
        X_pending: Optional[Tensor] = None,
        X_observed: Optional[Tensor] = None,
        eps: float = 1e-6,
        objective: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
    ) -> None:
        super().__init__(model=model)

        if reduction not in ("mean", "sum"):
            raise ValueError(f"Unknown reduction: {reduction}")

        self.ordinal_likelihood = _resolve_ordinal_likelihood(
            model=model,
            ordinal_likelihood=ordinal_likelihood,
        )
        self.sampler = sampler or SobolQMCNormalSampler(sample_shape=torch.Size([256]))
        self.reduction = reduction
        self.pending_penalty_weight = float(pending_penalty_weight)
        self.pending_penalty_beta = float(pending_penalty_beta)
        self.observed_penalty_weight = float(observed_penalty_weight)
        self.observed_penalty_beta = float(observed_penalty_beta)
        self.eps = float(eps)
        self.objective = objective

        self.X_pending: Optional[Tensor] = None
        self.set_X_pending(X_pending)

        self.X_observed: Optional[Tensor] = None
        self.set_X_observed(X_observed)

    def set_X_pending(self, X_pending: Optional[Tensor] = None) -> None:
        self.X_pending = _coerce_reference_to_tensor(X_pending)

    def set_X_observed(self, X_observed: Optional[Tensor] = None) -> None:
        self.X_observed = _coerce_reference_to_tensor(
            _resolve_observed_X(self.model, X_observed)
        )

    def _prepare_eval(self) -> None:
        self.model.eval()
        likelihood = getattr(self, "ordinal_likelihood", None)
        if hasattr(likelihood, "eval"):
            likelihood.eval()

    def _posterior(self, X: Tensor):
        return self.model.posterior(X)

    def _class_probs_from_posterior(self, X: Tensor) -> Tensor:
        posterior = self._posterior(X)
        probs = self.ordinal_likelihood.marginal_class_probs(posterior.distribution)
        return _reduce_probs_to_match_X(probs, X=X, eps=self.eps)

    def _reduce_q(self, score: Tensor) -> Tensor:
        if score.ndim == 0:
            return score

        if score.shape[-1] == 1:
            return score.squeeze(-1)

        if self.reduction == "mean":
            return score.mean(dim=-1)

        if self.reduction == "sum":
            return score.sum(dim=-1)

        raise ValueError(f"Unknown reduction: {self.reduction}")

    def _check_output_shape(self, out: Tensor, expected_batch_shape: torch.Size, name: str) -> None:
        if out.shape != expected_batch_shape:
            raise RuntimeError(
                f"{name}: output shape mismatch. "
                f"Expected {tuple(expected_batch_shape)}, got {tuple(out.shape)}."
            )

    def _pointwise_reference_penalty(self, Xt: Tensor) -> Tensor:
        penalty = torch.zeros(Xt.shape[:-1], device=Xt.device, dtype=Xt.dtype)

        if self.pending_penalty_weight > 0.0:
            Xp_t = _transform_reference_like_candidate(self.model, self.X_pending, ref=Xt)
            penalty = penalty + self.pending_penalty_weight * _rbf_reference_penalty_per_point(
                X=Xt,
                X_ref=Xp_t,
                beta=self.pending_penalty_beta,
            )

        if self.observed_penalty_weight > 0.0:
            Xobs_t = _transform_reference_like_candidate(self.model, self.X_observed, ref=Xt)
            penalty = penalty + self.observed_penalty_weight * _rbf_reference_penalty_per_point(
                X=Xt,
                X_ref=Xobs_t,
                beta=self.observed_penalty_beta,
            )

        return penalty

    def _finalize_pointwise_score(
        self,
        score: Tensor,
        X: Tensor,
        *,
        name: str,
        reduce_extra: Literal["mean", "sum"] = "mean",
    ) -> Tensor:
        raw_X = _ensure_q_batch(X)
        original_batch_shape = raw_X.shape[:-2]

        Xt = _apply_input_transform_for_reference(self.model, raw_X)

        score = _align_pointwise_score_to_X(
            score,
            Xt,
            name=f"{name} score before penalty",
            reduce_extra=reduce_extra,
        )

        penalty = self._pointwise_reference_penalty(Xt)
        if penalty.shape == score.shape:
            score = score - penalty
        else:
            penalty = _align_pointwise_score_to_X(
                penalty,
                Xt,
                name=f"{name} penalty",
                reduce_extra="sum",
            )
            score = score - penalty

        score = _align_pointwise_score_to_X(
            score,
            Xt,
            name=f"{name} score before objective",
            reduce_extra=reduce_extra,
        )

        score = _apply_ordinal_objective_to_pointwise_score(
            self,
            score,
            raw_X=raw_X,
            name=name,
        )

        out = self._reduce_q(score)
        self._check_output_shape(out, original_batch_shape, name)
        return out


_qOrdinalMCAcquisitionBase = _qOrdinalActiveLearningBase


# =========================================================
# Canonical active-learning acquisitions
# =========================================================
class qOrdinalPredictiveEntropy(_qOrdinalActiveLearningBase):
    """
    ordinal 用 predictive entropy acquisition。

    予測 class 分布の entropy が大きい点を選ぶ。
    classification 側の qBinaryPredictiveEntropy と同じく、
    pointwise score に penalty / objective を適用してから q 方向を reduce する。
    """

    @t_batch_mode_transform()
    @average_over_ensemble_models
    def forward(self, X: Tensor) -> Tensor:
        self._prepare_eval()
        posterior = self._posterior(X)
        probs = self.ordinal_likelihood.marginal_class_probs(posterior.distribution)
        probs = _reduce_probs_to_match_X(probs, X=X, eps=self.eps)

        score = ordinal_entropy_from_probs(probs, eps=self.eps)
        return self._finalize_pointwise_score(
            score,
            X,
            name="qOrdinalPredictiveEntropy",
        )


class qOrdinalBALD(_qOrdinalActiveLearningBase):
    """
    ordinal 用 BALD / mutual information acquisition。

    score = H[p(y|x,D)] - E_f[H[p(y|f)]]
    """

    def __init__(
        self,
        model: Model,
        ordinal_likelihood: Optional[OrdinalLogitLikelihood] = None,
        num_samples: int = 16,
        sampler: Optional[MCSampler] = None,
        reduction: ReductionType = "mean",
        pending_penalty_weight: float = 0.0,
        pending_penalty_beta: float = 10.0,
        observed_penalty_weight: float = 0.0,
        observed_penalty_beta: float = 10.0,
        X_pending: Optional[Tensor] = None,
        X_observed: Optional[Tensor] = None,
        eps: float = 1e-6,
        objective: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
    ) -> None:
        if sampler is None:
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([int(num_samples)]))
        super().__init__(
            model=model,
            ordinal_likelihood=ordinal_likelihood,
            sampler=sampler,
            reduction=reduction,
            pending_penalty_weight=pending_penalty_weight,
            pending_penalty_beta=pending_penalty_beta,
            observed_penalty_weight=observed_penalty_weight,
            observed_penalty_beta=observed_penalty_beta,
            X_pending=X_pending,
            X_observed=X_observed,
            eps=eps,
            objective=objective,
        )
        self.num_samples = int(num_samples)

    @t_batch_mode_transform()
    @average_over_ensemble_models
    def forward(self, X: Tensor) -> Tensor:
        self._prepare_eval()
        posterior = self._posterior(X)

        probs = self.ordinal_likelihood.marginal_class_probs(posterior.distribution)
        probs = _reduce_probs_to_match_X(probs, X=X, eps=self.eps)
        predictive_entropy = ordinal_entropy_from_probs(probs, eps=self.eps)

        latent_samples = self.get_posterior_samples(posterior).squeeze(-1)
        class_probs_given_f = self.ordinal_likelihood.class_probs_from_f(latent_samples)
        cond_entropy = ordinal_entropy_from_probs(class_probs_given_f, eps=self.eps)
        cond_entropy = _reduce_pointwise_scores_to_match_X(
            cond_entropy,
            X,
            reduce_extra="mean",
        )

        score = predictive_entropy - cond_entropy
        return self._finalize_pointwise_score(score, X, name="qOrdinalBALD")


def _default_utility_values(num_classes: int, *, device, dtype) -> Tensor:
    return torch.arange(num_classes, device=device, dtype=dtype)


def _utility_values_tensor(
    utility_values: Optional[Tensor],
    num_classes: int,
    *,
    device,
    dtype,
) -> Tensor:
    if utility_values is None:
        return _default_utility_values(num_classes, device=device, dtype=dtype)

    values = torch.as_tensor(utility_values, device=device, dtype=dtype).reshape(-1)
    if values.numel() != num_classes:
        raise ValueError(
            f"utility_values must have length {num_classes}, got {values.numel()}."
        )
    return values


class qOrdinalUtilityVariance(_qOrdinalActiveLearningBase):
    """
    ordinal 用の軽量 utility variance acquisition。

    class probability 上の utility 分散を使う proxy であり、mc_points は不要です。
    """

    def __init__(
        self,
        model: Model,
        ordinal_likelihood: Optional[OrdinalLogitLikelihood] = None,
        utility_values: Optional[Tensor] = None,
        sampler: Optional[MCSampler] = None,
        reduction: ReductionType = "mean",
        pending_penalty_weight: float = 0.0,
        pending_penalty_beta: float = 10.0,
        observed_penalty_weight: float = 0.0,
        observed_penalty_beta: float = 10.0,
        X_pending: Optional[Tensor] = None,
        X_observed: Optional[Tensor] = None,
        eps: float = 1e-6,
        objective: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
    ) -> None:
        super().__init__(
            model=model,
            ordinal_likelihood=ordinal_likelihood,
            sampler=sampler,
            reduction=reduction,
            pending_penalty_weight=pending_penalty_weight,
            pending_penalty_beta=pending_penalty_beta,
            observed_penalty_weight=observed_penalty_weight,
            observed_penalty_beta=observed_penalty_beta,
            X_pending=X_pending,
            X_observed=X_observed,
            eps=eps,
            objective=objective,
        )
        self.utility_values = utility_values

    @t_batch_mode_transform()
    @average_over_ensemble_models
    def forward(self, X: Tensor) -> Tensor:
        self._prepare_eval()
        probs = self._class_probs_from_posterior(X)

        utilities = _utility_values_tensor(
            self.utility_values,
            probs.shape[-1],
            device=probs.device,
            dtype=probs.dtype,
        )
        mean_u = (probs * utilities).sum(dim=-1)
        second_u = (probs * utilities.pow(2)).sum(dim=-1)
        score = (second_u - mean_u.pow(2)).clamp_min(0.0)

        return self._finalize_pointwise_score(
            score,
            X,
            name="qOrdinalUtilityVariance",
        )


class qOrdinalMarginUncertainty(_qOrdinalActiveLearningBase):
    """
    ordinal 用 margin uncertainty acquisition。

    top-1 class probability と top-2 class probability の差が小さい点を高く評価する。
    """

    @t_batch_mode_transform()
    @average_over_ensemble_models
    def forward(self, X: Tensor) -> Tensor:
        self._prepare_eval()
        probs = self._class_probs_from_posterior(X)

        top2 = torch.topk(probs, k=min(2, probs.shape[-1]), dim=-1).values
        if top2.shape[-1] == 1:
            score = torch.zeros_like(top2[..., 0])
        else:
            score = 1.0 - (top2[..., 0] - top2[..., 1])

        return self._finalize_pointwise_score(
            score,
            X,
            name="qOrdinalMarginUncertainty",
        )



# =========================================================
# Expensive / true-ish NIPV acquisition
# =========================================================
class qOrdinalFantasyNegIntegratedPosteriorVariance(AcquisitionFunction):
    """
    ordinal 用の fantasy negative integrated posterior variance。

    fantasy label を生成して condition_on_observations した後、
    mc_points 上の latent posterior variance を評価する高コスト版。

    Notes:
        この acquisition は score がすでに batch-level scalar なので、
        pointwise objective / q*n_w objective の適用対象としては通常使わない。
    """

    def __init__(
        self,
        model: Model,
        mc_points: Tensor,
        ordinal_likelihood: Optional[OrdinalLogitLikelihood] = None,
        num_fantasies: int = 8,
        conditioning_steps: int = 10,
        conditioning_lr: float | None = None,
        conditioning_batch_size: int | None = None,
        pending_penalty_weight: float = 0.0,
        pending_penalty_beta: float = 10.0,
        observed_penalty_weight: float = 0.0,
        observed_penalty_beta: float = 10.0,
        X_pending: Optional[Tensor] = None,
        X_observed: Optional[Tensor] = None,
        eps: float = 1e-6,
        objective: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
    ) -> None:
        super().__init__(model=model)

        if mc_points.ndim != 2:
            raise ValueError(
                f"mc_points must be [N_mc, d], got shape={tuple(mc_points.shape)}"
            )

        self.ordinal_likelihood = _resolve_ordinal_likelihood(
            model=model,
            ordinal_likelihood=ordinal_likelihood,
        )

        # train_X がない wrapper も想定して、mc_points 自身の dtype/device を基本にする。
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
        self.pending_penalty_weight = float(pending_penalty_weight)
        self.pending_penalty_beta = float(pending_penalty_beta)
        self.observed_penalty_weight = float(observed_penalty_weight)
        self.observed_penalty_beta = float(observed_penalty_beta)
        self.eps = float(eps)
        self.objective = objective

        self.X_pending: Optional[Tensor] = None
        self.set_X_pending(X_pending)

        self.X_observed: Optional[Tensor] = None
        self.set_X_observed(X_observed)

    def set_X_pending(self, X_pending: Optional[Tensor] = None) -> None:
        self.X_pending = _coerce_reference_to_tensor(X_pending)

    def set_X_observed(self, X_observed: Optional[Tensor] = None) -> None:
        self.X_observed = _coerce_reference_to_tensor(
            _resolve_observed_X(self.model, X_observed)
        )

    def _prepare_eval(self) -> None:
        self.model.eval()
        likelihood = getattr(self, "ordinal_likelihood", None)
        if hasattr(likelihood, "eval"):
            likelihood.eval()

    @torch.no_grad()
    def _sample_fantasy_labels(self, X: Tensor) -> Tensor:
        posterior = self.model.posterior(X)
        latent_samples = posterior.rsample(torch.Size([self.num_fantasies]))
        f = latent_samples.squeeze(-1)

        probs = self.ordinal_likelihood.class_probs_from_f(f)
        probs = probs.clamp_min(self.eps)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(self.eps)

        fantasy_Y = []
        for i in range(self.num_fantasies):
            yi = Categorical(probs=probs[i]).sample()
            fantasy_Y.append(yi.unsqueeze(-1))

        return torch.stack(fantasy_Y, dim=0)

    @torch.no_grad()
    def _integrated_latent_variance(self, fantasy_model: Model) -> Tensor:
        posterior = fantasy_model.posterior(self.mc_points)
        return posterior.variance.mean()

    def _aggregated_reference_penalty(self, X: Tensor) -> Tensor:
        Xt = _apply_input_transform_for_reference(self.model, X)
        penalty = torch.zeros(Xt.shape[:-2], device=Xt.device, dtype=Xt.dtype)

        if self.pending_penalty_weight > 0.0:
            Xp_t = _transform_reference_like_candidate(self.model, self.X_pending, ref=Xt)
            penalty = penalty + self.pending_penalty_weight * _rbf_reference_penalty_aggregated(
                X=Xt,
                X_ref=Xp_t,
                beta=self.pending_penalty_beta,
                reduction="sum",
            )

        if self.observed_penalty_weight > 0.0:
            Xobs_t = _transform_reference_like_candidate(self.model, self.X_observed, ref=Xt)
            penalty = penalty + self.observed_penalty_weight * _rbf_reference_penalty_aggregated(
                X=Xt,
                X_ref=Xobs_t,
                beta=self.observed_penalty_beta,
                reduction="sum",
            )

        return penalty

    def _apply_objective_to_scalar_score(self, score: Tensor, X: Tensor) -> Tensor:
        # scalar / batch-level score なので、InputPerturbation 用 objective は通常不要。
        # ただし weight/sign だけの objective などを使えるようにそのまま通す。
        return _apply_ordinal_objective_to_pointwise_score(
            self,
            score,
            raw_X=X,
            name="qOrdinalFantasyNegIntegratedPosteriorVariance",
        )

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._prepare_eval()
        X = _ensure_q_batch(X)
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
                    ipv = self._integrated_latent_variance(fantasy_model)
                vals.append(-ipv)

            out.append(torch.stack(vals).mean())

        score = torch.stack(out).reshape(*batch_shape)
        score = self._apply_objective_to_scalar_score(score, X)
        penalty = self._aggregated_reference_penalty(X)

        if penalty.shape != score.shape:
            if penalty.numel() == score.numel():
                penalty = penalty.reshape_as(score)
            else:
                raise RuntimeError(
                    "qOrdinalFantasyNegIntegratedPosteriorVariance penalty shape mismatch. "
                    f"score.shape={tuple(score.shape)}, penalty.shape={tuple(penalty.shape)}"
                )

        return score - penalty


__all__ = [
    "OrdinalScoreObjective",
    "_OrdinalScoreObjective",
    "qOrdinalPredictiveEntropy",
    "qOrdinalBALD",
    "qOrdinalUtilityVariance",
    "qOrdinalMarginUncertainty",
    "qOrdinalFantasyNegIntegratedPosteriorVariance",
]
