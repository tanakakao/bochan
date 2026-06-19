from __future__ import annotations

import math
from itertools import product
from typing import Any, Callable, Literal, Optional, Sequence

import torch
from torch import Tensor

from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.models.model import Model
from botorch.sampling.base import MCSampler
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.transforms import average_over_ensemble_models, t_batch_mode_transform


ReductionType = Literal["mean", "sum"]
QFeasMode = Literal["prod", "mean", "min", "max"]
OrdinalFeasibilityMode = Literal[
    "class_ge",
    "class_le",
    "class_interval",
    "expected_utility_ge",
]


# =========================================================
# Utilities for expected utility / best_f
# =========================================================
def _canonicalize_utility_values(
    utility_values: Sequence[float] | Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    utilities = torch.as_tensor(utility_values, device=device, dtype=dtype)
    if utilities.ndim != 1:
        raise ValueError(
            f"utility_values must be 1D. Got shape={tuple(utilities.shape)}."
        )
    return utilities


def compute_ordinal_expected_utility_values(
    model: Model,
    X: Tensor,
    utility_values: Sequence[float] | Tensor,
    maximize: bool = True,
) -> Tensor:
    """
    既存点の ordinal expected utility を計算する helper。

    Notes:
        best_f を計算する場合は、acquisition に渡す objective と同じ
        utility_values / maximize 設定を使ってください。
    """
    utilities = _canonicalize_utility_values(
        utility_values,
        device=X.device,
        dtype=X.dtype,
    )

    with torch.no_grad():
        if hasattr(model, "expected_utility"):
            values = model.expected_utility(X, utilities)
        elif hasattr(model, "class_probs"):
            probs = model.class_probs(X)
            if probs.shape[-1] != utilities.numel():
                raise RuntimeError(
                    "Number of classes in model.class_probs(X) does not match "
                    "utility_values."
                )
            values = (probs * utilities).sum(dim=-1)
        else:
            raise TypeError(
                "model must expose expected_utility(X, utilities) or class_probs(X)."
            )

        if values.ndim >= 1 and values.shape[-1] == 1:
            values = values.squeeze(-1)
        if not maximize:
            values = -values

    return values.detach()


def compute_ordinal_expected_utility_best_f(
    model: Model,
    train_X: Tensor,
    utility_values: Sequence[float] | Tensor,
    maximize: bool = True,
) -> Tensor:
    """train_X 上の expected utility から best_f を計算する helper。"""
    values = compute_ordinal_expected_utility_values(
        model=model,
        X=train_X,
        utility_values=utility_values,
        maximize=maximize,
    )
    return values.max().detach()


# =========================================================
# Tensor / pending helpers
# =========================================================
def ensure_q_batch(X: Tensor) -> Tensor:
    """
    X を `(..., q, d)` に揃える。

    - [d]    -> [1, 1, d]
    - [n, d] -> [1, n, d]
    - [b, q, d] 以上 -> そのまま
    """
    if not torch.is_tensor(X):
        raise TypeError(f"X must be Tensor. Got {type(X)}.")
    if X.ndim == 1:
        return X.view(1, 1, -1)
    if X.ndim == 2:
        return X.unsqueeze(0)
    return X


def _coerce_pending_to_tensor(
    X_pending,
    *,
    ref: Optional[Tensor] = None,
) -> Optional[Tensor]:
    """X_pending / X_observed を Tensor または None に正規化する。"""
    if X_pending is None:
        return None

    if torch.is_tensor(X_pending):
        out = X_pending
    elif isinstance(X_pending, (list, tuple)):
        tensors: list[Tensor] = []
        for item in X_pending:
            if item is None:
                continue
            t = _coerce_pending_to_tensor(item, ref=ref)
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
            f"Got {type(X_pending)}."
        )

    if ref is not None:
        out = out.to(device=ref.device, dtype=ref.dtype)
    return out


def _apply_input_transform_for_pending(model: Model, X: Tensor) -> Tensor:
    """candidate / pending / observed を同じ距離計算空間へ写す。"""
    X = ensure_q_batch(X)

    input_transform = getattr(model, "input_transform", None)
    if input_transform is not None:
        Xt = input_transform(X)
        if isinstance(Xt, tuple):
            Xt = Xt[0]
        return ensure_q_batch(Xt)

    models = getattr(model, "models", None)
    if models is not None and len(models) > 0:
        input_transform = getattr(models[0], "input_transform", None)
        if input_transform is not None:
            Xt = input_transform(X)
            if isinstance(Xt, tuple):
                Xt = Xt[0]
            return ensure_q_batch(Xt)

    return X


def _transform_reference_like_candidate(
    model: Model,
    X_ref,
    *,
    ref: Tensor,
) -> Optional[Tensor]:
    """raw-space の reference 点を candidate と同じ transformed space へ写す。"""
    Xr = _coerce_pending_to_tensor(X_ref, ref=ref)
    if Xr is None or Xr.numel() == 0:
        return None
    Xr_t = _apply_input_transform_for_pending(model, Xr)
    return Xr_t.to(device=ref.device, dtype=ref.dtype)


def _resolve_observed_X(model: Model, X_observed: Optional[Tensor] = None) -> Optional[Tensor]:
    """X_observed 省略時に model から train_X 相当を推定する。"""
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


def _flatten_points(X: Optional[Tensor]) -> Optional[Tensor]:
    if X is None or X.numel() == 0:
        return None
    return X.reshape(-1, X.shape[-1])


def _cat_dims_from_model(model: Model, d: int) -> list[int]:
    cat_dims = getattr(model, "cat_dims", [])
    try:
        return [int(j) for j in cat_dims if 0 <= int(j) < d]
    except TypeError:
        return []


def _pairwise_distance2(
    A: Tensor,
    B: Tensor,
    *,
    cat_dims: Optional[Sequence[int]] = None,
) -> Tensor:
    """
    mixed input を考慮した簡易距離。

    continuous dims: squared Euclidean
    categorical dims: mismatch count
    """
    d = A.shape[-1]
    cat_set = set(int(j) for j in (cat_dims or []) if 0 <= int(j) < d)
    cont_dims = [j for j in range(d) if j not in cat_set]
    cat_dims_valid = sorted(cat_set)

    dist2: Tensor | float = 0.0
    if len(cont_dims) > 0:
        A_cont = A[..., cont_dims]
        B_cont = B[..., cont_dims]
        dist2 = dist2 + (A_cont.unsqueeze(-2) - B_cont.unsqueeze(-3)).pow(2).sum(dim=-1)
    if len(cat_dims_valid) > 0:
        A_cat = A[..., cat_dims_valid]
        B_cat = B[..., cat_dims_valid]
        dist2 = dist2 + (A_cat.unsqueeze(-2) != B_cat.unsqueeze(-3)).to(A.dtype).sum(dim=-1)
    if isinstance(dist2, float):
        # d == 0 は通常あり得ないが、明示的に落とす。
        raise RuntimeError("No valid dimensions found for pairwise distance.")
    return dist2


def _reference_repulsion_penalty(
    X: Tensor,
    X_ref: Optional[Tensor],
    *,
    beta: float,
    weight: float,
    cat_dims: Optional[Sequence[int]] = None,
) -> Tensor:
    """X と reference 点の近接 penalty。戻り値は batch_shape。"""
    X = ensure_q_batch(X)
    batch_shape = X.shape[:-2]
    if weight <= 0.0:
        return X.new_zeros(batch_shape)

    X_ref_flat = _flatten_points(X_ref)
    if X_ref_flat is None:
        return X.new_zeros(batch_shape)

    d = X.shape[-1]
    Xb = X.reshape(-1, X.shape[-2], d)
    Xr = X_ref_flat.to(device=X.device, dtype=X.dtype)
    Xr = Xr.view(1, Xr.shape[0], d).expand(Xb.shape[0], -1, -1)

    d2 = _pairwise_distance2(Xb, Xr, cat_dims=cat_dims)
    nearest = d2.min(dim=-1).values  # [B, q]
    penalty = weight * torch.exp(-float(beta) * nearest).sum(dim=-1)
    return penalty.reshape(*batch_shape)


def _same_batch_repulsion_penalty(
    X: Tensor,
    *,
    beta: float,
    weight: float,
    cat_dims: Optional[Sequence[int]] = None,
) -> Tensor:
    """q-batch 内の候補点同士が近すぎる場合の penalty。戻り値は batch_shape。"""
    X = ensure_q_batch(X)
    batch_shape = X.shape[:-2]
    q = X.shape[-2]
    if q <= 1 or weight <= 0.0:
        return X.new_zeros(batch_shape)

    d = X.shape[-1]
    Xb = X.reshape(-1, q, d)
    d2 = _pairwise_distance2(Xb, Xb, cat_dims=cat_dims)
    eye = torch.eye(q, device=X.device, dtype=torch.bool).unsqueeze(0)
    d2 = d2.masked_fill(eye, float("inf"))

    # 対称ペアを二重カウントしないように 0.5 を掛ける。
    penalty = 0.5 * weight * torch.exp(-float(beta) * d2).sum(dim=(-1, -2))
    return penalty.reshape(*batch_shape)


def _sample_dims_from_sampler(sampler: Optional[MCSampler]) -> int:
    sample_shape = getattr(sampler, "sample_shape", torch.Size([1]))
    return len(sample_shape)


def _mean_over_sample_dims(values: Tensor, sampler: Optional[MCSampler]) -> Tensor:
    sample_ndim = _sample_dims_from_sampler(sampler)
    if sample_ndim <= 0:
        return values
    dims = tuple(range(sample_ndim))
    return values.mean(dim=dims)


def _std_over_sample_dims(values: Tensor, sampler: Optional[MCSampler]) -> Tensor:
    sample_ndim = _sample_dims_from_sampler(sampler)
    if sample_ndim <= 0:
        return torch.zeros_like(values)
    dims = tuple(range(sample_ndim))
    return values.std(dim=dims, unbiased=False)


def _normalize_utility_samples(
    utility_samples: Tensor,
    X: Tensor,
    *,
    sampler: Optional[MCSampler],
    name: str,
) -> Tensor:
    """
    utility objective の戻り値を `sample_shape x batch_shape x q_like` に揃える。

    InputPerturbation objective が q*n_w -> q を行う前提だが、未集約で
    q*n_w が返ってきた場合は安全側として mean で q に戻す。
    """
    Xq = ensure_q_batch(X)
    batch_shape = Xq.shape[:-2]
    q = Xq.shape[-2]

    # output dim=1 がある場合のみ squeeze。q=1 の q 次元は落とさない。
    if utility_samples.ndim > Xq.ndim and utility_samples.shape[-1] == 1:
        utility_samples = utility_samples.squeeze(-1)

    sample_ndim = _sample_dims_from_sampler(sampler)
    target_ndim = sample_ndim + len(batch_shape) + 1

    # Fully Bayesian / ensemble などの余分な batch dims は MC sample dims の直後で平均化。
    while utility_samples.ndim > target_ndim:
        reduce_dim = sample_ndim
        utility_samples = utility_samples.mean(dim=reduce_dim)

    expected_prefix = tuple(utility_samples.shape[:sample_ndim]) + tuple(batch_shape)
    if tuple(utility_samples.shape[: sample_ndim + len(batch_shape)]) != expected_prefix:
        # 最後の q_like 以外の要素数が一致する場合は reshape で救済。
        q_like = utility_samples.shape[-1]
        expected_numel = math.prod(utility_samples.shape[:sample_ndim]) * math.prod(batch_shape) * q_like
        if utility_samples.numel() == expected_numel:
            utility_samples = utility_samples.reshape(*utility_samples.shape[:sample_ndim], *batch_shape, q_like)

    q_like = utility_samples.shape[-1]
    if q_like == q:
        return utility_samples

    if q_like > q and q_like % q == 0:
        n_w = q_like // q
        return utility_samples.reshape(*utility_samples.shape[:-1], q, n_w).mean(dim=-1)

    raise RuntimeError(
        f"{name}: utility objective must return sample_shape x batch_shape x q "
        f"or q*n_w. X.shape={tuple(X.shape)}, "
        f"utility_samples.shape={tuple(utility_samples.shape)}."
    )


# =========================================================
# Ordinal likelihood / class probability helpers for PoF
# =========================================================
def _is_ordinal_likelihood(obj: Any) -> bool:
    return obj is not None and (
        hasattr(obj, "marginal_class_probs") or hasattr(obj, "class_probs_from_f")
    )


def _resolve_ordinal_likelihood(model: Model, ordinal_likelihood: Optional[Any] = None) -> Any:
    if ordinal_likelihood is not None:
        return ordinal_likelihood

    candidates: list[Any] = []
    candidates.append(getattr(model, "ordinal_likelihood", None))
    candidates.append(getattr(model, "likelihood", None))

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
            for cand in (getattr(m, "ordinal_likelihood", None), getattr(m, "likelihood", None)):
                if _is_ordinal_likelihood(cand):
                    likelihoods.append(cand)
        if len(likelihoods) == 1:
            return likelihoods[0]
        if len(likelihoods) > 1:
            raise ValueError(
                "Multiple ordinal likelihoods were found. Pass ordinal_likelihood explicitly."
            )

    raise ValueError(
        "ordinal_likelihood was not provided and could not be inferred from model."
    )


def _reduce_extra_batch_dims(tensor: Tensor, X: Tensor, n_trailing_keep: int) -> Tensor:
    out = tensor
    Xq = ensure_q_batch(X)
    x_batch_shape = tuple(Xq.shape[:-2])
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
        out = out.mean(dim=reduce_dim)
    return out


def _class_probs_from_model_or_likelihood(
    model: Model,
    X: Tensor,
    ordinal_likelihood: Any,
    eps: float,
) -> Tensor:
    """
    differentiable な ordinal class probability を返す。

    注意:
        acquisition optimization では X に対する gradient が必要なため、
        model.class_probs(X) のような予測用メソッドは原則使わない。
        class_probs が no_grad / detach されていると optimize_acqf で落ちる。
    """
    Xq = ensure_q_batch(X)

    posterior = model.posterior(Xq)

    # 1. likelihood が marginal_class_probs を持つ場合
    #    ただし、結果が X から切れている場合は fallback する。
    if hasattr(ordinal_likelihood, "marginal_class_probs"):
        try:
            probs = ordinal_likelihood.marginal_class_probs(posterior.distribution)
            probs = _reduce_extra_batch_dims(probs, X=Xq, n_trailing_keep=2)

            if probs.requires_grad or not Xq.requires_grad:
                probs = probs.clamp_min(eps)
                return probs / probs.sum(dim=-1, keepdim=True).clamp_min(eps)
        except Exception:
            pass

    # 2. fallback: posterior.mean を latent f とみなして class probability へ変換
    mean_f = posterior.mean

    if mean_f.ndim >= 1 and mean_f.shape[-1] == 1:
        mean_f = mean_f.squeeze(-1)

    mean_f = _reduce_extra_batch_dims(mean_f, X=Xq, n_trailing_keep=1)

    if hasattr(ordinal_likelihood, "class_probs_from_f"):
        probs = ordinal_likelihood.class_probs_from_f(mean_f)
    else:
        raise RuntimeError(
            "ordinal_likelihood must expose marginal_class_probs or class_probs_from_f "
            "for differentiable qOrdinalProbabilityOfFeasibility."
        )

    probs = _reduce_extra_batch_dims(probs, X=Xq, n_trailing_keep=2)
    probs = probs.clamp_min(eps)
    return probs / probs.sum(dim=-1, keepdim=True).clamp_min(eps)


# =========================================================
# Base class aligned with classification BO design
# =========================================================
class _OrdinalUtilityBOBase(MCAcquisitionFunction):
    """
    ordinal BO 用の共通 base。

    classification の `_BinaryProbabilityBOBase` と同じく、BoTorch 親クラスの
    default objective とユーザー指定 objective を混同しないように、
    ユーザー指定の utility objective は `utility_objective` として別管理する。
    """

    def __init__(
        self,
        model: Model,
        *,
        objective: Callable[[Tensor, Optional[Tensor]], Tensor],
        sampler: Optional[MCSampler] = None,
        eps: float = 1e-8,
        pending_penalty_weight: float = 0.0,
        pending_penalty_beta: float = 10.0,
        same_batch_penalty_weight: float = 0.0,
        same_batch_penalty_beta: float = 10.0,
        observed_penalty_weight: float = 0.0,
        observed_penalty_beta: float = 10.0,
        X_pending: Optional[Tensor] = None,
        X_observed: Optional[Tensor] = None,
        **kwargs,
    ) -> None:
        if sampler is None:
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))

        super().__init__(model=model, sampler=sampler, objective=None, **kwargs)

        if objective is None:
            raise ValueError("objective must be provided for ordinal utility BO acquisitions.")

        self.utility_objective = objective
        self.eps = float(eps)
        self.pending_penalty_weight = float(pending_penalty_weight)
        self.pending_penalty_beta = float(pending_penalty_beta)
        self.same_batch_penalty_weight = float(same_batch_penalty_weight)
        self.same_batch_penalty_beta = float(same_batch_penalty_beta)
        self.observed_penalty_weight = float(observed_penalty_weight)
        self.observed_penalty_beta = float(observed_penalty_beta)

        self.X_pending: Optional[Tensor] = None
        self.X_observed: Optional[Tensor] = None
        self.set_X_pending(X_pending)
        self.set_X_observed(X_observed)

    @property
    def cat_dims(self) -> list[int]:
        # input_transform 後も category dims が維持される設計を想定した簡易対応。
        # one-hot 等で次元が変わる transform の場合は Euclidean 距離として扱われる。
        return list(getattr(self.model, "cat_dims", []))

    def set_X_pending(self, X_pending: Optional[Tensor] = None) -> None:
        self.X_pending = _coerce_pending_to_tensor(X_pending)

    def set_X_observed(self, X_observed: Optional[Tensor] = None) -> None:
        resolved = _resolve_observed_X(self.model, X_observed)
        self.X_observed = _coerce_pending_to_tensor(resolved)

    def _posterior_samples_as_utility(self, X: Tensor, *, name: str) -> Tensor:
        Xq = ensure_q_batch(X)
        posterior = self.model.posterior(Xq)
        samples = self.get_posterior_samples(posterior)

        try:
            utility = self.utility_objective(samples, X=Xq)
        except TypeError:
            utility = self.utility_objective(samples)

        if not torch.is_tensor(utility):
            raise RuntimeError(f"{name}: objective must return Tensor. Got {type(utility)}.")

        return _normalize_utility_samples(
            utility,
            Xq,
            sampler=self.sampler,
            name=name,
        )

    def _repulsion_penalty(self, X: Tensor) -> Tensor:
        Xt = _apply_input_transform_for_pending(self.model, X)
        d = Xt.shape[-1]
        cat_dims = _cat_dims_from_model(self.model, d)

        penalty = _same_batch_repulsion_penalty(
            Xt,
            beta=self.same_batch_penalty_beta,
            weight=self.same_batch_penalty_weight,
            cat_dims=cat_dims,
        )

        Xp_t = _transform_reference_like_candidate(self.model, self.X_pending, ref=Xt)
        penalty = penalty + _reference_repulsion_penalty(
            Xt,
            Xp_t,
            beta=self.pending_penalty_beta,
            weight=self.pending_penalty_weight,
            cat_dims=cat_dims,
        )

        Xobs_t = _transform_reference_like_candidate(self.model, self.X_observed, ref=Xt)
        penalty = penalty + _reference_repulsion_penalty(
            Xt,
            Xobs_t,
            beta=self.observed_penalty_beta,
            weight=self.observed_penalty_weight,
            cat_dims=cat_dims,
        )

        return penalty

    def _apply_repulsion_penalty(self, X: Tensor, value: Tensor) -> Tensor:
        penalty = self._repulsion_penalty(X).to(device=value.device, dtype=value.dtype)
        return value - penalty


# =========================================================
# Canonical ordinal BO acquisitions
# =========================================================
class qOrdinalExpectedUtility(_OrdinalUtilityBOBase):
    """ordinal class を utility に変換し、q-batch 内の最大 utility の期待値を最大化する。"""

    @t_batch_mode_transform()
    @average_over_ensemble_models
    def forward(self, X: Tensor) -> Tensor:
        utility_samples = self._posterior_samples_as_utility(
            X,
            name="qOrdinalExpectedUtility",
        )
        value = utility_samples.max(dim=-1).values
        out = _mean_over_sample_dims(value, self.sampler)
        return self._apply_repulsion_penalty(X, out)


class qOrdinalExpectedImprovement(_OrdinalUtilityBOBase):
    """expected utility 上の q-Expected Improvement。"""

    def __init__(self, model: Model, best_f: float | Tensor, **kwargs) -> None:
        super().__init__(model=model, **kwargs)
        self.register_buffer("best_f", torch.as_tensor(best_f))

    @t_batch_mode_transform()
    @average_over_ensemble_models
    def forward(self, X: Tensor) -> Tensor:
        utility_samples = self._posterior_samples_as_utility(
            X,
            name="qOrdinalExpectedImprovement",
        )
        best_q = utility_samples.max(dim=-1).values
        best_f = self.best_f.to(device=best_q.device, dtype=best_q.dtype)
        improvement = (best_q - best_f).clamp_min(0.0)
        out = _mean_over_sample_dims(improvement, self.sampler)
        return self._apply_repulsion_penalty(X, out)


class qOrdinalProbabilityOfImprovement(_OrdinalUtilityBOBase):
    """expected utility 上の soft q-Probability of Improvement。"""

    def __init__(
        self,
        model: Model,
        best_f: float | Tensor,
        tau: float = 1e-3,
        **kwargs,
    ) -> None:
        super().__init__(model=model, **kwargs)
        self.register_buffer("best_f", torch.as_tensor(best_f))
        self.register_buffer("tau", torch.as_tensor(tau))

    @t_batch_mode_transform()
    @average_over_ensemble_models
    def forward(self, X: Tensor) -> Tensor:
        utility_samples = self._posterior_samples_as_utility(
            X,
            name="qOrdinalProbabilityOfImprovement",
        )
        best_q = utility_samples.max(dim=-1).values
        best_f = self.best_f.to(device=best_q.device, dtype=best_q.dtype)
        tau = self.tau.to(device=best_q.device, dtype=best_q.dtype).clamp_min(1e-9)
        prob_improvement = torch.sigmoid((best_q - best_f) / tau)
        out = _mean_over_sample_dims(prob_improvement, self.sampler)
        return self._apply_repulsion_penalty(X, out)


class qOrdinalUpperConfidenceBound(_OrdinalUtilityBOBase):
    """expected utility samples から mean + sqrt(beta) * std を計算する UCB。"""

    def __init__(self, model: Model, beta: float | Tensor = 2.0, **kwargs) -> None:
        super().__init__(model=model, **kwargs)
        self.register_buffer("beta", torch.as_tensor(beta))

    @t_batch_mode_transform()
    @average_over_ensemble_models
    def forward(self, X: Tensor) -> Tensor:
        utility_samples = self._posterior_samples_as_utility(
            X,
            name="qOrdinalUpperConfidenceBound",
        )
        mean = _mean_over_sample_dims(utility_samples, self.sampler)
        std = _std_over_sample_dims(utility_samples, self.sampler).clamp_min(self.eps)
        beta = self.beta.to(device=mean.device, dtype=mean.dtype)
        score = mean + beta.sqrt() * std
        out = score.max(dim=-1).values
        return self._apply_repulsion_penalty(X, out)


# =========================================================
# Optional ordinal Probability of Feasibility
# =========================================================
class qOrdinalProbabilityOfFeasibility(AcquisitionFunction):
    """
    ordinal constraint 用 Probability of Feasibility。

    Modes:
        - class_ge: P(y >= min_class)
        - class_le: P(y <= max_class)
        - class_interval: P(min_class <= y <= max_class)
        - expected_utility_ge: E[u(y)] >= utility_threshold の soft 確率 proxy

    Notes:
        BO の目的側 acquisition に feasibility weight として掛ける場合や、
        ordinal constraint 単独で feasible region を探索したい場合に使う。
    """

    def __init__(
        self,
        model: Model,
        ordinal_likelihood: Optional[Any] = None,
        mode: OrdinalFeasibilityMode = "class_ge",
        min_class: Optional[int] = None,
        max_class: Optional[int] = None,
        utility_values: Optional[Sequence[float] | Tensor] = None,
        utility_threshold: Optional[float | Tensor] = None,
        tau: float = 1e-3,
        q_feas_mode: QFeasMode = "prod",
        eps: float = 1e-8,
        pending_penalty_weight: float = 0.0,
        pending_penalty_beta: float = 10.0,
        same_batch_penalty_weight: float = 0.0,
        same_batch_penalty_beta: float = 10.0,
        X_pending: Optional[Tensor] = None,
    ) -> None:
        super().__init__(model=model)
        self.ordinal_likelihood = _resolve_ordinal_likelihood(model, ordinal_likelihood)
        self.mode = mode
        self.min_class = min_class
        self.max_class = max_class
        self.utility_values = utility_values
        self.utility_threshold = utility_threshold
        self.tau = float(tau)
        self.q_feas_mode = q_feas_mode
        self.eps = float(eps)
        self.pending_penalty_weight = float(pending_penalty_weight)
        self.pending_penalty_beta = float(pending_penalty_beta)
        self.same_batch_penalty_weight = float(same_batch_penalty_weight)
        self.same_batch_penalty_beta = float(same_batch_penalty_beta)
        self.X_pending: Optional[Tensor] = None
        self.set_X_pending(X_pending)

    def set_X_pending(self, X_pending: Optional[Tensor] = None) -> None:
        self.X_pending = _coerce_pending_to_tensor(X_pending)

    def _pointwise_feasibility(self, X: Tensor) -> Tensor:
        probs = _class_probs_from_model_or_likelihood(
            self.model,
            X,
            self.ordinal_likelihood,
            eps=self.eps,
        )
        num_classes = probs.shape[-1]

        if self.mode == "class_ge":
            if self.min_class is None:
                raise ValueError("min_class must be specified for mode='class_ge'.")
            k = int(self.min_class)
            if not (0 <= k < num_classes):
                raise ValueError(f"min_class must be in [0, {num_classes - 1}].")
            return probs[..., k:].sum(dim=-1)

        if self.mode == "class_le":
            if self.max_class is None:
                raise ValueError("max_class must be specified for mode='class_le'.")
            k = int(self.max_class)
            if not (0 <= k < num_classes):
                raise ValueError(f"max_class must be in [0, {num_classes - 1}].")
            return probs[..., : k + 1].sum(dim=-1)

        if self.mode == "class_interval":
            if self.min_class is None or self.max_class is None:
                raise ValueError(
                    "min_class and max_class must be specified for mode='class_interval'."
                )
            lo = int(self.min_class)
            hi = int(self.max_class)
            if lo > hi:
                raise ValueError("min_class must be <= max_class.")
            if not (0 <= lo < num_classes and 0 <= hi < num_classes):
                raise ValueError(f"class bounds must be in [0, {num_classes - 1}].")
            return probs[..., lo : hi + 1].sum(dim=-1)

        if self.mode == "expected_utility_ge":
            if self.utility_values is None:
                raise ValueError(
                    "utility_values must be specified for mode='expected_utility_ge'."
                )
            if self.utility_threshold is None:
                raise ValueError(
                    "utility_threshold must be specified for mode='expected_utility_ge'."
                )
            utilities = _canonicalize_utility_values(
                self.utility_values,
                device=probs.device,
                dtype=probs.dtype,
            )
            if utilities.numel() != num_classes:
                raise ValueError(
                    f"utility_values must have length {num_classes}, got {utilities.numel()}."
                )
            expected_u = (probs * utilities).sum(dim=-1)
            threshold = torch.as_tensor(
                self.utility_threshold,
                device=expected_u.device,
                dtype=expected_u.dtype,
            )
            tau = torch.as_tensor(self.tau, device=expected_u.device, dtype=expected_u.dtype)
            return torch.sigmoid((expected_u - threshold) / tau.clamp_min(1e-9))

        raise ValueError(f"Unknown ordinal feasibility mode: {self.mode}")

    def _reduce_q_feasibility(self, point_feas: Tensor) -> Tensor:
        point_feas = point_feas.clamp(self.eps, 1.0 - self.eps)
        if self.q_feas_mode == "prod":
            return point_feas.prod(dim=-1)
        if self.q_feas_mode == "mean":
            return point_feas.mean(dim=-1)
        if self.q_feas_mode == "min":
            return point_feas.min(dim=-1).values
        if self.q_feas_mode == "max":
            return point_feas.max(dim=-1).values
        raise ValueError(f"Unknown q_feas_mode: {self.q_feas_mode}")

    def _repulsion_penalty(self, X: Tensor, value: Tensor) -> Tensor:
        Xt = _apply_input_transform_for_pending(self.model, X)
        d = Xt.shape[-1]
        cat_dims = _cat_dims_from_model(self.model, d)
        penalty = _same_batch_repulsion_penalty(
            Xt,
            beta=self.same_batch_penalty_beta,
            weight=self.same_batch_penalty_weight,
            cat_dims=cat_dims,
        )
        Xp_t = _transform_reference_like_candidate(self.model, self.X_pending, ref=Xt)
        penalty = penalty + _reference_repulsion_penalty(
            Xt,
            Xp_t,
            beta=self.pending_penalty_beta,
            weight=self.pending_penalty_weight,
            cat_dims=cat_dims,
        )
        return value - penalty.to(device=value.device, dtype=value.dtype)

    @t_batch_mode_transform()
    @average_over_ensemble_models
    def forward(self, X: Tensor) -> Tensor:
        point_feas = self._pointwise_feasibility(X)
        value = self._reduce_q_feasibility(point_feas)
        return self._repulsion_penalty(X, value)


# =========================================================
# Mixed categorical helper
# =========================================================
def make_fixed_features_list(
    cat_dims: Sequence[int],
    category_counts: dict[int, int],
) -> list[dict[int, float]]:
    cat_dims = sorted(int(j) for j in cat_dims)
    grids = [range(int(category_counts[j])) for j in cat_dims]
    return [{j: float(v) for j, v in zip(cat_dims, vals)} for vals in product(*grids)]


__all__ = [
    "compute_ordinal_expected_utility_values",
    "compute_ordinal_expected_utility_best_f",
    "qOrdinalExpectedUtility",
    "qOrdinalExpectedImprovement",
    "qOrdinalProbabilityOfImprovement",
    "qOrdinalUpperConfidenceBound",
    "qOrdinalProbabilityOfFeasibility",
    "make_fixed_features_list",
]
