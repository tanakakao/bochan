from __future__ import annotations

import math
from typing import Callable, Literal, Optional, Sequence

import torch
from torch import Tensor

from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.models.model import Model
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.transforms import t_batch_mode_transform

from ..hetero_utils import get_hetero_ordinal_summary, get_noise_sigma


RiskType = Optional[Literal["var", "cvar"]]
ReductionType = Literal["mean", "sum", "max"]
NoiseWeightMode = Literal["none", "inverse_linear", "inverse_exp", "custom"]
NoiseCombineType = Literal["multiply", "subtract"]
ROIWeightMode = Literal["none", "utility_above", "utility_below", "utility_interval", "custom"]
ROICombineType = Literal["multiply", "add"]


# =========================================================
# Score objective
# =========================================================
class HeteroOrdinalScoreObjective(torch.nn.Module):
    """
    hetero ordinal active learning / level-set の pointwise score に作用する objective。

    posterior samples ではなく、acquisition 内で計算済みの score に作用する。
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
                    "HeteroOrdinalScoreObjective received an aggregated score. "
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


# Backward-compatible internal name used by older examples.
_HeteroOrdinalScoreObjective = HeteroOrdinalScoreObjective


# =========================================================
# Generic helpers
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


def _default_ordinal_likelihood(model: Model):
    lk = getattr(model, "ordinal_likelihood", None)
    if lk is not None:
        return lk
    lk = getattr(model, "likelihood", None)
    if lk is not None:
        return lk
    return None


def _entropy_from_probs(probs: Tensor, eps: float = 1e-12) -> Tensor:
    probs = probs.clamp_min(eps)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(eps)
    return -(probs * probs.log()).sum(dim=-1)


def _ordinal_logit_latent_to_probs(
    latent_samples: Tensor,
    ordinal_likelihood,
    eps: float = 1e-12,
) -> Tensor:
    f = latent_samples
    if f.ndim >= 1 and f.shape[-1] == 1:
        f = f.squeeze(-1)

    if ordinal_likelihood is None:
        raise ValueError("ordinal_likelihood is required to convert latent samples to probabilities.")

    num_classes = int(getattr(ordinal_likelihood, "num_classes", 0))
    cutpoints = None
    for name in ("cutpoints", "transformed_cutpoints", "thresholds", "_cutpoints"):
        obj = getattr(ordinal_likelihood, name, None)
        if obj is not None:
            cutpoints = obj() if callable(obj) else obj
            break

    if cutpoints is None:
        raw_cutpoints = getattr(ordinal_likelihood, "raw_cutpoints", None)
        if raw_cutpoints is not None:
            if hasattr(ordinal_likelihood, "_ordered_cutpoints"):
                cutpoints = ordinal_likelihood._ordered_cutpoints()
            elif hasattr(ordinal_likelihood, "transform_cutpoints"):
                cutpoints = ordinal_likelihood.transform_cutpoints(raw_cutpoints)

    if cutpoints is None:
        raise ValueError("Could not obtain cutpoints from ordinal_likelihood.")

    cutpoints = torch.as_tensor(
        cutpoints,
        device=f.device,
        dtype=f.dtype,
    ).detach().flatten()

    if num_classes <= 0:
        num_classes = int(cutpoints.numel() + 1)

    z = cutpoints.view(*((1,) * f.ndim), -1) - f.unsqueeze(-1)
    cdf = torch.sigmoid(z)

    p0 = cdf[..., :1]
    plast = 1.0 - cdf[..., -1:]
    if num_classes == 2:
        probs = torch.cat([p0, plast], dim=-1)
    else:
        pmid = cdf[..., 1:] - cdf[..., :-1]
        probs = torch.cat([p0, pmid, plast], dim=-1)

    probs = probs.clamp_min(eps)
    return probs / probs.sum(dim=-1, keepdim=True).clamp_min(eps)


def _latent_samples_to_probs(
    latent_samples: Tensor,
    *,
    ordinal_likelihood=None,
    latent_to_probs: Optional[Callable[[Tensor], Tensor]] = None,
    eps: float = 1e-12,
) -> Tensor:
    if latent_samples.ndim >= 1 and latent_samples.shape[-1] != 1:
        latent_in = latent_samples.unsqueeze(-1)
    else:
        latent_in = latent_samples

    if latent_to_probs is not None:
        probs = latent_to_probs(latent_in)
    else:
        probs = None
        if ordinal_likelihood is not None:
            for name in (
                "probs_from_latent",
                "class_probs_from_latent",
                "class_probs_from_f",
                "latent_to_probs",
                "probs",
            ):
                fn = getattr(ordinal_likelihood, name, None)
                if callable(fn):
                    probs = fn(latent_in)
                    break

            if probs is None:
                probs = _ordinal_logit_latent_to_probs(
                    latent_in,
                    ordinal_likelihood=ordinal_likelihood,
                    eps=eps,
                )

        if probs is None:
            raise ValueError(
                "Could not convert latent samples to class probabilities. "
                "Pass latent_to_probs explicitly or provide cutpoints in ordinal_likelihood."
            )

    if probs.ndim >= 2 and probs.shape[-2] == 1:
        probs = probs.squeeze(-2)

    probs = probs.clamp_min(eps)
    return probs / probs.sum(dim=-1, keepdim=True).clamp_min(eps)


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

    # trailing dims が target と一致するまで extra dims を reduce
    while out.ndim > len(target):
        out = out.mean(dim=0) if reduce_extra == "mean" else out.sum(dim=0)
        if out.shape == target:
            return out

    if out.shape == target:
        return out

    if out.ndim == len(target) and out.shape[-1] == target[-1]:
        # batch dims の broadcast を試す
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


def _apply_hetero_ordinal_objective_to_score(
    owner,
    score: Tensor,
    X: Optional[Tensor] = None,
    name: str = "HeteroOrdinalActiveLearning",
) -> Tensor:
    objective = getattr(owner, "objective", None)
    if objective is None:
        return score

    out = _objective_call(objective, score, X)

    if not torch.is_tensor(out):
        raise RuntimeError(f"{name}: objective must return a Tensor. Got {type(out)}.")

    return out


# =========================================================
# Classification-aligned hetero ordinal base
# =========================================================
class _BaseHeteroOrdinalActiveLearningAcquisition(MCAcquisitionFunction):
    """
    classification hetero single acquisition に寄せた ordinal hetero base。

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
        *,
        utility_values: Optional[Sequence[float] | Tensor] = None,
        reduction: ReductionType = "mean",
        # backward-compatible alias. If provided, overrides reduction.
        reduce: Optional[str] = None,
        eps: float = 1e-12,
        sampler: Optional[SobolQMCNormalSampler] = None,
        # pending penalty
        pending_penalty_weight: float = 0.0,
        pending_penalty_beta: float = 10.0,
        X_pending: Optional[Tensor] = None,
        # ROI weighting based on expected utility if available in summary
        roi_mode: ROIWeightMode = "none",
        roi_combine: ROICombineType = "multiply",
        roi_threshold: float = 0.5,
        roi_interval: Optional[tuple[float, float]] = None,
        roi_beta: float = 20.0,
        roi_bandwidth: float = 0.15,
        roi_min_weight: float = 0.0,
        roi_weight_scale: float = 1.0,
        roi_weight_fn: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
        # noise weighting
        noise_mode: NoiseWeightMode = "inverse_linear",
        noise_combine: NoiseCombineType = "multiply",
        noise_penalty_lambda: float = 1.0,
        noise_min_weight: float = 0.0,
        noise_weight_scale: float = 1.0,
        noise_weight_fn: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
        # compatibility with older ordinal implementation
        noise_penalty: Optional[float] = None,
        default_sigma: float = 0.0,
        objective: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
    ) -> None:
        if sampler is None:
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))
        super().__init__(model=model, sampler=sampler)

        if reduce is not None:
            reduction = str(reduce)  # old API accepted "max".
        if reduction not in ("mean", "sum", "max"):
            raise ValueError("reduction must be 'mean', 'sum', or 'max'.")

        if roi_mode not in ("none", "utility_above", "utility_below", "utility_interval", "custom"):
            raise ValueError(f"Unknown roi_mode: {roi_mode!r}.")
        if roi_combine not in ("multiply", "add"):
            raise ValueError("roi_combine must be 'multiply' or 'add'.")
        if noise_mode not in ("none", "inverse_linear", "inverse_exp", "custom"):
            raise ValueError(f"Unknown noise_mode: {noise_mode!r}.")
        if noise_combine not in ("multiply", "subtract"):
            raise ValueError("noise_combine must be 'multiply' or 'subtract'.")

        self.utility_values = utility_values
        self.reduction = reduction
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

        # old `noise_penalty` means subtract sigma penalty. Preserve it.
        if noise_penalty is not None:
            noise_combine = "subtract"
            noise_penalty_lambda = float(noise_penalty)
            if noise_mode == "inverse_linear":
                noise_mode = "none"

        self.noise_mode = noise_mode
        self.noise_combine = noise_combine
        self.noise_penalty_lambda = float(noise_penalty_lambda)
        self.noise_min_weight = float(noise_min_weight)
        self.noise_weight_scale = float(noise_weight_scale)
        self.noise_weight_fn = noise_weight_fn
        self.default_sigma = float(default_sigma)

        self.X_pending: Optional[Tensor] = None
        self.set_X_pending(X_pending)

    # BoTorch optimize_acqf sequential=True requires set_X_pending.
    def set_X_pending(self, X_pending: Optional[Tensor] = None) -> None:
        self.X_pending = _coerce_reference_to_tensor(X_pending)

    def _prepare_eval(self) -> None:
        self.model.eval()
        likelihood = getattr(self.model, "likelihood", None)
        if likelihood is not None and hasattr(likelihood, "eval"):
            likelihood.eval()
        ordinal_likelihood = getattr(self.model, "ordinal_likelihood", None)
        if ordinal_likelihood is not None and hasattr(ordinal_likelihood, "eval"):
            ordinal_likelihood.eval()

    def _apply_input_transform(self, X: Tensor) -> Tensor:
        X = _ensure_q_batch(X)
        it = getattr(self.model, "input_transform", None)
        if it is None:
            return X
        Xt = it(X)
        if isinstance(Xt, tuple):
            Xt = Xt[0]
        return _ensure_q_batch(Xt)

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
            default_sigma=self.default_sigma,
            eps=self.eps,
        )

    def _get_roi_signal(self, summary: dict[str, Tensor], score: Tensor) -> Tensor:
        # Prefer expected utility-like quantities if hetero_utils exposes them.
        for key in ("mean_u", "expected_u", "utility_mean", "mean_utility"):
            if key in summary and torch.is_tensor(summary[key]):
                return summary[key]
        return score

    def _roi_weight_per_point(
        self,
        signal: Tensor,
        X: Optional[Tensor],
    ) -> Tensor:
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

        if self.roi_mode == "utility_above":
            w = torch.sigmoid(self.roi_beta * (signal - self.roi_threshold))
        elif self.roi_mode == "utility_below":
            w = torch.sigmoid(self.roi_beta * (self.roi_threshold - signal))
        elif self.roi_mode == "utility_interval":
            if self.roi_interval is None:
                raise ValueError("roi_interval must be provided when roi_mode='utility_interval'.")
            lo, hi = self.roi_interval
            w_lo = torch.sigmoid(self.roi_beta * (signal - float(lo)))
            w_hi = torch.sigmoid(self.roi_beta * (float(hi) - signal))
            w = w_lo * w_hi
        else:
            raise ValueError(f"Unknown roi_mode: {self.roi_mode!r}.")

        if self.roi_bandwidth > 0.0 and self.roi_mode in ("utility_above", "utility_below"):
            # Smooth band around threshold; keeps weight from becoming too binary.
            band = torch.exp(-0.5 * ((signal - self.roi_threshold) / self.roi_bandwidth) ** 2)
            w = 0.5 * w + 0.5 * band

        w = self.roi_min_weight + self.roi_weight_scale * w
        return w.clamp_min(self.roi_min_weight)

    def _apply_roi_weight_per_point(
        self,
        score: Tensor,
        signal: Tensor,
        X: Optional[Tensor],
    ) -> Tensor:
        if self.roi_mode == "none":
            return score
        w = self._roi_weight_per_point(signal, X)
        w = _align_pointwise_score_to_X(w, _ensure_q_batch(X), name="ROI weight", reduce_extra="mean") if X is not None and w.shape != score.shape else w
        if self.roi_combine == "multiply":
            return score * w
        if self.roi_combine == "add":
            return score + w
        raise ValueError(f"Unknown roi_combine: {self.roi_combine!r}.")

    def _noise_weight_per_point(
        self,
        sigma: Tensor,
        X: Optional[Tensor],
    ) -> Tensor:
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

    def _apply_noise_weight_per_point(
        self,
        score: Tensor,
        sigma: Tensor,
        X: Optional[Tensor],
    ) -> Tensor:
        sigma = _align_pointwise_score_to_X(
            sigma,
            _ensure_q_batch(X) if X is not None else sigma.unsqueeze(-1),
            name="noise sigma",
            reduce_extra="mean",
        ) if sigma.shape != score.shape and X is not None else sigma

        if self.noise_combine == "subtract":
            return score - self.noise_penalty_lambda * sigma

        if self.noise_combine == "multiply":
            if self.noise_mode == "none":
                return score
            return score * self._noise_weight_per_point(sigma, X)

        raise ValueError(f"Unknown noise_combine: {self.noise_combine!r}.")

    def _apply_objective_to_score(self, score: Tensor, X: Tensor, name: str) -> Tensor:
        return _apply_hetero_ordinal_objective_to_score(self, score, X=X, name=name)

    def _finalize_pointwise_score(
        self,
        score: Tensor,
        X: Tensor,
        *,
        summary: Optional[dict[str, Tensor]] = None,
        roi_signal: Optional[Tensor] = None,
        name: str,
    ) -> Tensor:
        raw_X = _ensure_q_batch(X)
        original_batch_shape = raw_X.shape[:-2]
        Xt = self._apply_input_transform(raw_X)

        score = _align_pointwise_score_to_X(
            score,
            Xt,
            name=f"{name} score before weighting",
            reduce_extra="mean",
        )

        if summary is not None:
            if roi_signal is None:
                roi_signal = self._get_roi_signal(summary, score)
            roi_signal = _align_pointwise_score_to_X(
                roi_signal,
                Xt,
                name=f"{name} ROI signal",
                reduce_extra="mean",
            )
            score = self._apply_roi_weight_per_point(score, roi_signal, Xt)

            if "sigma" in summary:
                sigma = summary["sigma"]
            else:
                sigma = get_noise_sigma(
                    self.model,
                    raw_X,
                    mean_like=score.unsqueeze(-1),
                    default_sigma=self.default_sigma,
                ).squeeze(-1)
            sigma = _align_pointwise_score_to_X(
                sigma,
                Xt,
                name=f"{name} sigma",
                reduce_extra="mean",
            )
            score = self._apply_noise_weight_per_point(score, sigma, Xt)

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
# Acquisition implementations
# =========================================================
class qHeteroOrdinalPredictiveEntropy(_BaseHeteroOrdinalActiveLearningAcquisition):
    """heteroscedastic ordinal predictive entropy acquisition。

    ordinal class probability の entropy を pointwise score として使います。
    """

    def __init__(self, model: Model, **kwargs) -> None:
        super().__init__(model=model, **kwargs)

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._prepare_eval()
        summary = self._summary(_ensure_q_batch(X))
        score = summary["entropy"]
        return self._finalize_pointwise_score(
            score,
            X,
            summary=summary,
            name="qHeteroOrdinalPredictiveEntropy",
        )


class qHeteroOrdinalUtilityVariance(_BaseHeteroOrdinalActiveLearningAcquisition):
    """heteroscedastic ordinal utility variance acquisition。

    expected utility の分散を pointwise score として使います。
    """

    def __init__(self, model: Model, **kwargs) -> None:
        super().__init__(model=model, **kwargs)

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._prepare_eval()
        summary = self._summary(_ensure_q_batch(X))
        score = summary["var_u"]
        return self._finalize_pointwise_score(
            score,
            X,
            summary=summary,
            name="qHeteroOrdinalUtilityVariance",
        )


class qHeteroOrdinalMarginUncertainty(_BaseHeteroOrdinalActiveLearningAcquisition):
    """heteroscedastic ordinal margin uncertainty acquisition。

    class probability の top-2 margin 由来の不確実性を pointwise score として使います。
    """

    def __init__(self, model: Model, **kwargs) -> None:
        super().__init__(model=model, **kwargs)

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._prepare_eval()
        summary = self._summary(_ensure_q_batch(X))
        score = summary["margin_uncertainty"]
        return self._finalize_pointwise_score(
            score,
            X,
            summary=summary,
            name="qHeteroOrdinalMarginUncertainty",
        )


class qHeteroOrdinalBALD(_BaseHeteroOrdinalActiveLearningAcquisition):
    r"""
    heteroscedastic ordinal BALD / mutual-information acquisition。

    BALD(x) = H[p(y|x,D)] - E_{f ~ p(f|x,D)}[H[p(y|f)]]
    """

    def __init__(
        self,
        model: Model,
        *,
        ordinal_likelihood=None,
        latent_to_probs: Optional[Callable[[Tensor], Tensor]] = None,
        num_samples: int = 128,
        **kwargs,
    ) -> None:
        sampler = kwargs.pop("sampler", None)
        if sampler is None:
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([int(num_samples)]))
        super().__init__(model=model, sampler=sampler, **kwargs)
        self.num_samples = int(num_samples)
        self.ordinal_likelihood = (
            ordinal_likelihood
            if ordinal_likelihood is not None
            else _default_ordinal_likelihood(model)
        )
        self.latent_to_probs = latent_to_probs

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._prepare_eval()
        raw_X = _ensure_q_batch(X)

        posterior = self.model.posterior(raw_X, observation_noise=False)
        latent_samples = self.get_posterior_samples(posterior)

        probs_mc = _latent_samples_to_probs(
            latent_samples,
            ordinal_likelihood=self.ordinal_likelihood,
            latent_to_probs=self.latent_to_probs,
            eps=self.eps,
        )

        pred_probs = probs_mc.mean(dim=0)
        pred_ent = _entropy_from_probs(pred_probs, eps=self.eps)
        cond_ent = _entropy_from_probs(probs_mc, eps=self.eps).mean(dim=0)
        score = pred_ent - cond_ent

        # For BALD we still use summary for sigma / ROI signal if available.
        try:
            summary = self._summary(raw_X)
        except Exception:
            sigma = get_noise_sigma(
                self.model,
                raw_X,
                mean_like=score.unsqueeze(-1),
                default_sigma=self.default_sigma,
            ).squeeze(-1)
            summary = {"sigma": sigma}

        return self._finalize_pointwise_score(
            score,
            raw_X,
            summary=summary,
            name="qHeteroOrdinalBALD",
        )


# =========================================================
# Public exports
# =========================================================
# 元ファイルで公開されていた import 名は維持:
#   qHeteroOrdinalPredictiveEntropy
#   qHeteroOrdinalUtilityVariance
#   qHeteroOrdinalMarginUncertainty
#   qHeteroOrdinalBALD
#   qHeteroOrdinalIntegratedPosteriorVariance
#
# 不要な compatibility alias は定義しない。
class qHeteroOrdinalIntegratedPosteriorVariance(qHeteroOrdinalUtilityVariance):
    """
    Lightweight IPV-style proxy based on hetero noise-aware ordinal utility variance.

    This is a proxy, not true/fantasy integrated posterior variance.
    """
    pass


__all__ = [
    "HeteroOrdinalScoreObjective",
    "_HeteroOrdinalScoreObjective",
    "qHeteroOrdinalPredictiveEntropy",
    "qHeteroOrdinalUtilityVariance",
    "qHeteroOrdinalMarginUncertainty",
    "qHeteroOrdinalBALD",
    "qHeteroOrdinalIntegratedPosteriorVariance",
]
