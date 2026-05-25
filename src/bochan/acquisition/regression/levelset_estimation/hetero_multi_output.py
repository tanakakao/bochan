from __future__ import annotations

from typing import Any, Callable, Literal, Optional, Sequence

import torch
from torch import Tensor

from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.models.model import Model
from botorch.utils.transforms import t_batch_mode_transform


ReductionType = Literal["mean", "sum", "max", "min"]
OutputReductionType = Literal[
    "mean",
    "sum",
    "max",
    "min",
    "weighted_sum",
    "weighted_mean",
]
VarianceSource = Literal["latent", "total", "noise"]
NoiseCombineType = Literal["subtract", "multiply", "none"]
NoiseWeightMode = Literal["inverse_linear", "inverse_sqrt", "exp", "none"]


def _reduce(t: Tensor, dim: int, mode: str) -> Tensor:
    if mode == "mean":
        return t.mean(dim=dim)
    if mode == "sum":
        return t.sum(dim=dim)
    if mode == "max":
        return t.max(dim=dim).values
    if mode == "min":
        return t.min(dim=dim).values
    raise ValueError(f"Unknown reduction mode: {mode!r}.")


def _ensure_q_batch(X: Tensor) -> Tensor:
    if not torch.is_tensor(X):
        raise TypeError(f"X must be a Tensor. Got {type(X)}.")
    if X.ndim == 1:
        return X.view(1, 1, -1)
    if X.ndim == 2:
        return X.unsqueeze(0)
    return X


def _safe_prod(shape: torch.Size | tuple[int, ...]) -> int:
    out = 1
    for s in shape:
        out *= int(s)
    return out


def _objective_call(objective: Callable, score: Tensor, X: Optional[Tensor]):
    try:
        return objective(score, X=X)
    except TypeError:
        return objective(score)


def _safe_normal_cdf(z: Tensor) -> Tensor:
    two = torch.as_tensor(2.0, device=z.device, dtype=z.dtype)
    return 0.5 * (1.0 + torch.erf(z / torch.sqrt(two)))


def _safe_logdet(covar: Tensor, jitter: float = 1e-6) -> Tensor:
    q = covar.shape[-1]
    eye = torch.eye(q, device=covar.device, dtype=covar.dtype)
    while eye.ndim < covar.ndim:
        eye = eye.unsqueeze(0)
    covar = 0.5 * (covar + covar.transpose(-1, -2))
    return torch.linalg.slogdet(covar + jitter * eye).logabsdet


BoundaryMode = Literal[
    "distance_to_threshold",
    "common_satisfaction",
    "all_above",
    "all_below",
]
ProbabilityMode = Literal["above", "below", "interval"]


class HeteroMultiOutputRegressionLevelSetScoreObjective(torch.nn.Module):
    """Objective applied to hetero multi-output regression level-set scores."""

    def __init__(
        self,
        n_w: Optional[int] = None,
        risk_type: Optional[Literal["var", "cvar"]] = None,
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
    def _is_aggregated_score(score: Tensor, X: Optional[Tensor]) -> bool:
        if X is None or score.ndim == 0:
            return False
        Xq = _ensure_q_batch(X)
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
                    "HeteroMultiOutputRegressionLevelSetScoreObjective received an aggregated score. "
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

        descending = not self.maximize
        sorted_score = torch.sort(score_w, dim=-1, descending=descending).values
        k = max(1, int(torch.ceil(torch.as_tensor(self.n_w * self.alpha)).item()))
        tail = sorted_score[..., :k]

        if self.risk_type == "var":
            return tail[..., -1]
        if self.risk_type == "cvar":
            return tail.mean(dim=-1)
        raise ValueError(f"Unknown risk_type: {self.risk_type!r}.")


# The level-set base intentionally duplicates the active-learning base essentials
# so this file can be used independently.
class _HeteroMultiOutputRegressionLevelSetBase(AcquisitionFunction):
    """Noise-aware multi-output regression level-set base."""

    def __init__(
        self,
        model: Model,
        *,
        thresholds: Optional[Sequence[float] | Tensor] = None,
        threshold: Optional[float | Tensor] = None,
        h: Optional[Sequence[float] | Tensor] = None,
        reduction: ReductionType = "mean",
        output_reduction: OutputReductionType = "weighted_mean",
        output_weights: Optional[Tensor | Sequence[float]] = None,
        normalize_output_weights: bool = True,
        variance_source: VarianceSource = "latent",
        noise_penalty: Optional[float] = None,
        noise_penalty_lambda: float = 1.0,
        noise_mode: NoiseWeightMode = "inverse_linear",
        noise_combine: NoiseCombineType = "subtract",
        noise_min_weight: float = 0.0,
        X_pending: Optional[Tensor] = None,
        X_observed: Optional[Tensor] = None,
        same_batch_penalty_weight: float = 0.0,
        same_batch_penalty_beta: float = 10.0,
        pending_penalty_weight: float = 0.0,
        pending_penalty_beta: float = 10.0,
        observed_penalty_weight: float = 0.0,
        observed_penalty_beta: float = 10.0,
        hard_duplicate_penalty: float = 0.0,
        hard_duplicate_tol: float = 1e-8,
        objective: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
        n_w: Optional[int] = None,
        eps: float = 1e-12,
    ) -> None:
        super().__init__(model=model)

        if h is not None:
            thresholds = h
        if thresholds is None:
            thresholds = threshold if threshold is not None else 0.0

        if reduction not in ("mean", "sum", "max", "min"):
            raise ValueError("reduction must be one of 'mean', 'sum', 'max', 'min'.")
        if output_reduction not in (
            "mean",
            "sum",
            "max",
            "min",
            "weighted_sum",
            "weighted_mean",
        ):
            raise ValueError(
                "output_reduction must be one of "
                "'mean', 'sum', 'max', 'min', 'weighted_sum', 'weighted_mean'."
            )
        if variance_source not in ("latent", "total", "noise"):
            raise ValueError("variance_source must be 'latent', 'total', or 'noise'.")
        if noise_mode not in ("inverse_linear", "inverse_sqrt", "exp", "none"):
            raise ValueError("noise_mode must be 'inverse_linear', 'inverse_sqrt', 'exp', or 'none'.")
        if noise_combine not in ("subtract", "multiply", "none"):
            raise ValueError("noise_combine must be 'subtract', 'multiply', or 'none'.")

        self.register_buffer("thresholds", torch.as_tensor(thresholds).reshape(-1))
        self.reduction = reduction
        self.output_reduction = output_reduction
        self.normalize_output_weights = bool(normalize_output_weights)

        if output_weights is not None:
            w = torch.as_tensor(output_weights)
            if w.ndim != 1:
                raise ValueError("output_weights must have shape [m].")
            self.register_buffer("output_weights", w.detach().clone())
        else:
            self.output_weights = None

        self.variance_source = variance_source
        if noise_penalty is not None:
            noise_penalty_lambda = float(noise_penalty)
            noise_combine = "subtract"
        self.noise_penalty_lambda = float(noise_penalty_lambda)
        self.noise_mode = noise_mode
        self.noise_combine = noise_combine
        self.noise_min_weight = float(noise_min_weight)

        self.same_batch_penalty_weight = float(same_batch_penalty_weight)
        self.same_batch_penalty_beta = float(same_batch_penalty_beta)
        self.pending_penalty_weight = float(pending_penalty_weight)
        self.pending_penalty_beta = float(pending_penalty_beta)
        self.observed_penalty_weight = float(observed_penalty_weight)
        self.observed_penalty_beta = float(observed_penalty_beta)
        self.hard_duplicate_penalty = float(hard_duplicate_penalty)
        self.hard_duplicate_tol = float(hard_duplicate_tol)

        self.objective = objective
        self.eps = float(eps)
        if n_w is None and objective is not None:
            n_w = getattr(objective, "n_w", None)
        self.n_w = None if n_w is None else int(n_w)

        self.X_pending: Optional[Tensor] = None
        self.X_observed: Optional[Tensor] = None
        self.set_X_pending(X_pending)
        self.set_X_observed(X_observed)

    def _coerce_reference_to_tensor(self, ref, *, like: Optional[Tensor] = None) -> Optional[Tensor]:
        if ref is None:
            return None
        if torch.is_tensor(ref):
            out = ref
        elif isinstance(ref, (list, tuple)):
            tensors = []
            for item in ref:
                if item is None:
                    continue
                t = self._coerce_reference_to_tensor(item, like=like)
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
                "Reference points must be None, Tensor, list, or tuple. "
                f"Got {type(ref)}."
            )
        if like is not None:
            out = out.to(device=like.device, dtype=like.dtype)
        return out.detach()

    def set_X_pending(self, X_pending: Optional[Tensor] = None) -> None:
        self.X_pending = self._coerce_reference_to_tensor(X_pending)

    def set_X_observed(self, X_observed: Optional[Tensor] = None) -> None:
        self.X_observed = self._coerce_reference_to_tensor(X_observed)

    def _prepare_eval(self) -> None:
        self.model.eval()
        likelihood = getattr(self.model, "likelihood", None)
        if likelihood is not None and hasattr(likelihood, "eval"):
            likelihood.eval()

    def _apply_input_transform_for_distance(self, X: Tensor) -> Tensor:
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

    def _reference_to_distance_space(self, ref, *, like: Tensor) -> Optional[Tensor]:
        ref = self._coerce_reference_to_tensor(ref, like=like)
        if ref is None or ref.numel() == 0:
            return None
        ref_t = self._apply_input_transform_for_distance(ref)
        return _ensure_q_batch(ref_t).to(device=like.device, dtype=like.dtype)

    def _output_weights_like(self, value: Tensor) -> Optional[Tensor]:
        weights = self.output_weights
        if weights is None:
            return None
        if value.shape[-1] != weights.numel():
            raise ValueError(
                f"Mismatch between output dim {value.shape[-1]} and output_weights {weights.numel()}."
            )
        w = weights.to(device=value.device, dtype=value.dtype)
        if self.normalize_output_weights:
            w = w / w.sum().clamp_min(self.eps)
        return w

    def _reduce_outputs(self, value: Tensor) -> Tensor:
        if value.shape[-1] == 1:
            return value.squeeze(-1)
        if self.output_reduction == "weighted_sum":
            w = self._output_weights_like(value)
            if w is None:
                raise ValueError("output_reduction='weighted_sum' requires output_weights.")
            return (value * w).sum(dim=-1)
        if self.output_reduction == "weighted_mean":
            w = self._output_weights_like(value)
            if w is None:
                return value.mean(dim=-1)
            return (value * w).sum(dim=-1)
        return _reduce(value, dim=-1, mode=self.output_reduction)

    def _thresholds_like(self, value: Tensor) -> Tensor:
        m = int(value.shape[-1])
        thresholds = self.thresholds.to(device=value.device, dtype=value.dtype)
        if thresholds.numel() == 1:
            thresholds = thresholds.expand(m)
        elif thresholds.numel() != m:
            raise ValueError(
                f"Number of thresholds ({thresholds.numel()}) does not match output dim ({m})."
            )
        return thresholds.view(*((1,) * (value.ndim - 1)), m)

    def _align_output_tensor_to_X(self, value: Tensor, Xt: Tensor, *, name: str) -> Tensor:
        Xt = _ensure_q_batch(Xt)
        target_prefix = torch.Size(Xt.shape[:-1])
        out = value
        if out.shape == target_prefix:
            return out.unsqueeze(-1)
        while out.ndim > len(target_prefix) + 1:
            out = out.mean(dim=0)
            if out.shape == target_prefix:
                return out.unsqueeze(-1)
        if out.ndim == len(target_prefix) + 1 and out.shape[:-1] == target_prefix:
            return out
        if out.ndim == len(target_prefix) and out.shape == target_prefix:
            return out.unsqueeze(-1)
        if out.numel() % max(_safe_prod(target_prefix), 1) == 0:
            m = out.numel() // max(_safe_prod(target_prefix), 1)
            return out.reshape(*target_prefix, m)
        raise RuntimeError(
            f"{name}: could not align tensor to output shape. "
            f"value.shape={tuple(value.shape)}, Xt.shape={tuple(Xt.shape)}."
        )

    def _align_pointwise_score_to_X(
        self,
        score: Tensor,
        Xt: Tensor,
        *,
        name: str,
        reduce_extra: ReductionType = "mean",
    ) -> Tensor:
        Xt = _ensure_q_batch(Xt)
        target = torch.Size(Xt.shape[:-1])
        out = score
        if out.shape == target:
            return out
        if out.ndim >= 1 and out.shape[-1] == 1:
            out_s = out.squeeze(-1)
            if out_s.shape == target:
                return out_s
            out = out_s
        if out.shape == target:
            return out
        while out.ndim > len(target):
            out = _reduce(out, dim=0, mode=reduce_extra)
            if out.shape == target:
                return out
        if out.shape == target:
            return out
        if out.numel() == _safe_prod(target):
            return out.reshape(target)
        raise RuntimeError(
            f"{name}: score shape mismatch. "
            f"score.shape={tuple(score.shape)}, expected={tuple(target)}, Xt.shape={tuple(Xt.shape)}."
        )

    def _posterior_mean_variance_outputs(self, X: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        Xq = _ensure_q_batch(X)
        self._prepare_eval()
        try:
            post_latent = self.model.posterior(Xq, observation_noise=False)
            post_total = self.model.posterior(Xq, observation_noise=True)
        except Exception:
            post_latent = self.model.posterior(Xq)
            post_total = post_latent
        Xt = self._apply_input_transform_for_distance(Xq)
        mean = self._align_output_tensor_to_X(post_latent.mean, Xt, name="posterior.mean")
        latent_var = self._align_output_tensor_to_X(post_latent.variance, Xt, name="latent variance")
        total_var = self._align_output_tensor_to_X(post_total.variance, Xt, name="total variance")
        latent_var = latent_var.clamp_min(self.eps)
        total_var = total_var.clamp_min(self.eps)
        noise_var = (total_var - latent_var).clamp_min(self.eps)

        noise_fn = getattr(self.model, "predict_noise_var", None)
        if callable(noise_fn):
            try:
                noise_raw = noise_fn(Xq)
                noise_var = self._align_output_tensor_to_X(noise_raw, Xt, name="predict_noise_var")
                noise_var = noise_var.clamp_min(self.eps)
                total_var = (latent_var + noise_var).clamp_min(self.eps)
            except Exception:
                pass
        return mean, latent_var, total_var, noise_var, Xt

    def _select_variance_outputs(
        self,
        latent_var: Tensor,
        total_var: Tensor,
        noise_var: Tensor,
    ) -> Tensor:
        if self.variance_source == "latent":
            return latent_var
        if self.variance_source == "total":
            return total_var
        if self.variance_source == "noise":
            return noise_var
        raise ValueError(f"Unknown variance_source: {self.variance_source!r}.")

    def _noise_weight_outputs(self, noise_var: Tensor) -> Tensor:
        lam = self.noise_penalty_lambda
        if self.noise_mode == "none":
            weight = torch.ones_like(noise_var)
        elif self.noise_mode == "inverse_linear":
            weight = 1.0 / (1.0 + lam * noise_var)
        elif self.noise_mode == "inverse_sqrt":
            weight = 1.0 / torch.sqrt(1.0 + lam * noise_var)
        elif self.noise_mode == "exp":
            weight = torch.exp(-lam * noise_var)
        else:
            raise ValueError(f"Unknown noise_mode={self.noise_mode!r}.")
        if self.noise_min_weight > 0.0:
            weight = self.noise_min_weight + (1.0 - self.noise_min_weight) * weight
        return weight

    def _apply_noise_to_output_score(self, score_outputs: Tensor, noise_var_outputs: Tensor) -> Tensor:
        if self.noise_combine == "none":
            return score_outputs
        if self.noise_combine == "subtract":
            return score_outputs - self.noise_penalty_lambda * noise_var_outputs
        if self.noise_combine == "multiply":
            return score_outputs * self._noise_weight_outputs(noise_var_outputs)
        raise ValueError(f"Unknown noise_combine={self.noise_combine!r}.")

    def _posterior_covariance(self, X: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        mean_outputs, latent_var_outputs, _, _, Xt = self._posterior_mean_variance_outputs(X)
        mean = self._reduce_outputs(mean_outputs)
        latent_var = self._reduce_outputs(latent_var_outputs)
        posterior = self.model.posterior(_ensure_q_batch(X), observation_noise=False)

        covar = None
        mvn = getattr(posterior, "mvn", None)
        if mvn is not None and hasattr(mvn, "covariance_matrix"):
            covar = mvn.covariance_matrix
        elif hasattr(posterior, "distribution") and hasattr(posterior.distribution, "covariance_matrix"):
            covar = posterior.distribution.covariance_matrix

        q_like = int(Xt.shape[-2])
        target_covar_shape = torch.Size(Xt.shape[:-2]) + torch.Size([q_like, q_like])
        if covar is None:
            return mean, torch.diag_embed(latent_var), Xt
        while covar.ndim > len(target_covar_shape):
            covar = covar.mean(dim=0)
            if covar.shape == target_covar_shape:
                break
        if covar.shape != target_covar_shape:
            if covar.numel() == _safe_prod(target_covar_shape):
                covar = covar.reshape(target_covar_shape)
            else:
                covar = torch.diag_embed(latent_var)
        covar = 0.5 * (covar + covar.transpose(-1, -2))
        return mean, covar, Xt

    def _same_batch_penalty_per_point(self, Xt: Tensor) -> Tensor:
        Xt = _ensure_q_batch(Xt)
        q = int(Xt.shape[-2])
        if self.same_batch_penalty_weight <= 0.0 or q <= 1:
            return Xt.new_zeros(Xt.shape[:-1])
        d2 = (Xt.unsqueeze(-2) - Xt.unsqueeze(-3)).pow(2).sum(dim=-1)
        eye = torch.eye(q, dtype=torch.bool, device=Xt.device)
        while eye.ndim < d2.ndim:
            eye = eye.unsqueeze(0)
        valid = ~eye
        soft = torch.exp(-self.same_batch_penalty_beta * d2)
        soft = torch.where(valid, soft, torch.zeros_like(soft))
        per_point = soft.sum(dim=-1)
        if self.hard_duplicate_penalty > 0.0:
            dup = (d2 <= self.hard_duplicate_tol).to(dtype=Xt.dtype)
            dup = torch.where(valid, dup, torch.zeros_like(dup))
            per_point = per_point + self.hard_duplicate_penalty * dup.sum(dim=-1)
        return self.same_batch_penalty_weight * per_point

    def _reference_penalty_per_point(self, Xt: Tensor, ref, *, weight: float, beta: float) -> Tensor:
        Xt = _ensure_q_batch(Xt)
        if weight <= 0.0:
            return Xt.new_zeros(Xt.shape[:-1])
        ref_t = self._reference_to_distance_space(ref, like=Xt)
        if ref_t is None or ref_t.numel() == 0:
            return Xt.new_zeros(Xt.shape[:-1])
        ref2d = ref_t.reshape(-1, ref_t.shape[-1])
        if ref2d.shape[-1] != Xt.shape[-1]:
            raise RuntimeError(
                "Reference feature dimension mismatch after transform: "
                f"Xt.shape={tuple(Xt.shape)}, ref_transformed.shape={tuple(ref_t.shape)}."
            )
        dist = torch.cdist(Xt.reshape(-1, Xt.shape[-1]), ref2d)
        min_dist = dist.min(dim=-1).values.reshape(*Xt.shape[:-1])
        return weight * torch.exp(-beta * min_dist)

    def _total_penalty_per_point(self, Xt: Tensor) -> Tensor:
        return (
            self._same_batch_penalty_per_point(Xt)
            + self._reference_penalty_per_point(
                Xt, self.X_pending, weight=self.pending_penalty_weight, beta=self.pending_penalty_beta
            )
            + self._reference_penalty_per_point(
                Xt, self.X_observed, weight=self.observed_penalty_weight, beta=self.observed_penalty_beta
            )
        )

    def _apply_objective_to_score(self, score: Tensor, X: Tensor, name: str) -> Tensor:
        if self.objective is None:
            return score
        out = _objective_call(self.objective, score, X)
        if not torch.is_tensor(out):
            raise RuntimeError(f"{name}: objective must return Tensor. Got {type(out)}.")
        return out

    def _aggregate_n_w_if_needed(self, score: Tensor, *, q: int, context: str) -> Tensor:
        if self.n_w is None:
            return score
        expected = q * int(self.n_w)
        if score.shape[-1] == q:
            return score
        if score.shape[-1] != expected:
            raise RuntimeError(
                f"{context}: expected last dimension q={q} or q*n_w={expected}, "
                f"got score.shape={tuple(score.shape)}."
            )
        return score.reshape(*score.shape[:-1], q, int(self.n_w)).mean(dim=-1)

    def _reduce_q(self, score: Tensor) -> Tensor:
        return _reduce(score, dim=-1, mode=self.reduction)

    def _finalize_pointwise_score(self, score: Tensor, X: Tensor, Xt: Tensor, *, name: str) -> Tensor:
        raw_X = _ensure_q_batch(X)
        original_batch_shape = torch.Size(raw_X.shape[:-2])
        q = int(raw_X.shape[-2])
        score = self._align_pointwise_score_to_X(score, Xt, name=f"{name} score before penalty")
        score = score - self._total_penalty_per_point(Xt)
        score = self._align_pointwise_score_to_X(score, Xt, name=f"{name} score before objective")
        score = self._apply_objective_to_score(score, raw_X, name=name)
        score = self._aggregate_n_w_if_needed(score, q=q, context=name)
        out = self._reduce_q(score)
        if out.shape == original_batch_shape:
            return out
        while out.ndim > len(original_batch_shape):
            out = out.mean(dim=0)
            if out.shape == original_batch_shape:
                return out
        if out.shape == original_batch_shape:
            return out
        if out.numel() == _safe_prod(original_batch_shape):
            return out.reshape(original_batch_shape)
        raise RuntimeError(
            f"{name}: output shape mismatch. "
            f"Expected {tuple(original_batch_shape)}, got {tuple(out.shape)}."
        )

    def _finalize_joint_score(self, score: Tensor, X: Tensor, Xt: Tensor, *, name: str) -> Tensor:
        raw_X = _ensure_q_batch(X)
        original_batch_shape = torch.Size(raw_X.shape[:-2])
        if score.shape != Xt.shape[:-2]:
            while score.ndim > len(Xt.shape[:-2]):
                score = score.mean(dim=0)
            if score.shape != Xt.shape[:-2] and score.numel() == _safe_prod(Xt.shape[:-2]):
                score = score.reshape(Xt.shape[:-2])
            if score.shape != Xt.shape[:-2]:
                raise RuntimeError(
                    f"{name}: joint score shape mismatch. "
                    f"score.shape={tuple(score.shape)}, expected={tuple(Xt.shape[:-2])}."
                )
        penalty = self._total_penalty_per_point(Xt)
        penalty = self._reduce_q(penalty)
        score = score - penalty
        score = self._apply_objective_to_score(score, raw_X, name=name)
        if score.shape == original_batch_shape:
            return score
        while score.ndim > len(original_batch_shape):
            score = score.mean(dim=0)
        if score.shape == original_batch_shape:
            return score
        if score.numel() == _safe_prod(original_batch_shape):
            return score.reshape(original_batch_shape)
        raise RuntimeError(
            f"{name}: output shape mismatch. "
            f"Expected {tuple(original_batch_shape)}, got {tuple(score.shape)}."
        )


class qHeteroMultiOutputRegressionStraddle(_HeteroMultiOutputRegressionLevelSetBase):
    """Noise-aware multi-output regression straddle acquisition."""

    def __init__(
        self,
        model: Model,
        *,
        beta: float | Tensor = 1.96,
        boundary_mode: BoundaryMode = "distance_to_threshold",
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, **kwargs)
        if boundary_mode not in (
            "distance_to_threshold", "common_satisfaction", "all_above", "all_below"
        ):
            raise ValueError(
                "boundary_mode must be 'distance_to_threshold', "
                "'common_satisfaction', 'all_above', or 'all_below'."
            )
        self.register_buffer("beta", torch.as_tensor(beta))
        self.boundary_mode = boundary_mode

    def _boundary_distance(self, mean: Tensor, thresholds: Tensor) -> Tensor:
        if self.boundary_mode == "distance_to_threshold":
            return (mean - thresholds).abs()
        if self.boundary_mode in ("common_satisfaction", "all_above"):
            return torch.relu(thresholds - mean)
        if self.boundary_mode == "all_below":
            return torch.relu(mean - thresholds)
        raise ValueError(f"Unknown boundary_mode: {self.boundary_mode!r}.")

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        mean, latent_var, total_var, noise_var, Xt = self._posterior_mean_variance_outputs(X)
        var = self._select_variance_outputs(latent_var, total_var, noise_var)
        thresholds = self._thresholds_like(mean)
        beta = self.beta.to(device=mean.device, dtype=mean.dtype)
        score_outputs = beta * var.sqrt() - self._boundary_distance(mean, thresholds)
        score_outputs = self._apply_noise_to_output_score(score_outputs, noise_var)
        score = self._reduce_outputs(score_outputs)
        return self._finalize_pointwise_score(score, X, Xt, name="qHeteroMultiOutputRegressionStraddle")


class qHeteroMultiOutputRegressionJointStraddle(_HeteroMultiOutputRegressionLevelSetBase):
    """Joint noise-aware multi-output regression straddle acquisition."""

    def __init__(
        self,
        model: Model,
        *,
        beta: float | Tensor = 1.0,
        uncertainty_measure: Literal["logdet", "logdet1p", "trace"] = "logdet1p",
        boundary_mode: BoundaryMode = "distance_to_threshold",
        covariance_jitter: float = 1e-6,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, **kwargs)
        if uncertainty_measure not in ("logdet", "logdet1p", "trace"):
            raise ValueError("uncertainty_measure must be 'logdet', 'logdet1p', or 'trace'.")
        self.register_buffer("beta", torch.as_tensor(beta))
        self.uncertainty_measure = uncertainty_measure
        self.boundary_mode = boundary_mode
        self.covariance_jitter = float(covariance_jitter)

    def _boundary_distance(self, mean_outputs: Tensor, thresholds: Tensor) -> Tensor:
        if self.boundary_mode == "distance_to_threshold":
            return (mean_outputs - thresholds).abs()
        if self.boundary_mode in ("common_satisfaction", "all_above"):
            return torch.relu(thresholds - mean_outputs)
        if self.boundary_mode == "all_below":
            return torch.relu(mean_outputs - thresholds)
        raise ValueError(f"Unknown boundary_mode: {self.boundary_mode!r}.")

    def _uncertainty_score(self, covar: Tensor) -> Tensor:
        if self.uncertainty_measure == "trace":
            return covar.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        if self.uncertainty_measure == "logdet":
            return _safe_logdet(covar, jitter=self.covariance_jitter)
        q = covar.shape[-1]
        eye = torch.eye(q, device=covar.device, dtype=covar.dtype)
        while eye.ndim < covar.ndim:
            eye = eye.unsqueeze(0)
        return _safe_logdet(eye + covar, jitter=self.covariance_jitter)

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        mean_outputs, _, _, noise_var_outputs, Xt0 = self._posterior_mean_variance_outputs(X)
        thresholds = self._thresholds_like(mean_outputs)
        _, covar, Xt = self._posterior_covariance(X)
        beta = self.beta.to(device=mean_outputs.device, dtype=mean_outputs.dtype)
        boundary = self._boundary_distance(mean_outputs, thresholds)
        boundary_score = -self._reduce_outputs(boundary).mean(dim=-1)
        uncertainty = self._uncertainty_score(covar)
        noise_score = self._reduce_outputs(noise_var_outputs).mean(dim=-1)
        if self.noise_combine == "subtract":
            score = boundary_score + beta * uncertainty - self.noise_penalty_lambda * noise_score
        elif self.noise_combine == "multiply":
            score = (boundary_score + beta * uncertainty) * self._noise_weight_outputs(noise_score)
        else:
            score = boundary_score + beta * uncertainty
        return self._finalize_joint_score(score, X, Xt, name="qHeteroMultiOutputRegressionJointStraddle")


class qHeteroMultiOutputRegressionICU(_HeteroMultiOutputRegressionLevelSetBase):
    """Noise-aware multi-output integrated contour uncertainty acquisition."""

    def __init__(
        self,
        model: Model,
        *,
        bandwidth: Optional[float | Tensor] = None,
        joint_boundary: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, **kwargs)
        self.bandwidth = None if bandwidth is None else torch.as_tensor(bandwidth)
        self.joint_boundary = bool(joint_boundary)

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        mean, latent_var, total_var, noise_var, Xt = self._posterior_mean_variance_outputs(X)
        var = self._select_variance_outputs(latent_var, total_var, noise_var)
        std = var.sqrt().clamp_min(self.eps)
        thresholds = self._thresholds_like(mean)
        if self.bandwidth is None:
            bw = std
        else:
            bw = self.bandwidth.to(device=mean.device, dtype=mean.dtype).clamp_min(self.eps)
        z = (mean - thresholds) / bw
        score_outputs = torch.exp(-0.5 * z.pow(2)) * std
        score_outputs = self._apply_noise_to_output_score(score_outputs, noise_var)
        score = score_outputs.prod(dim=-1) if self.joint_boundary else self._reduce_outputs(score_outputs)
        return self._finalize_pointwise_score(score, X, Xt, name="qHeteroMultiOutputRegressionICU")


class qHeteroMultiOutputRegressionBoundaryVariance(_HeteroMultiOutputRegressionLevelSetBase):
    """Noise-aware boundary-weighted variance acquisition."""

    def __init__(
        self,
        model: Model,
        *,
        tau: float | Tensor = 1.0,
        joint_boundary: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, **kwargs)
        self.register_buffer("tau", torch.as_tensor(tau))
        self.joint_boundary = bool(joint_boundary)

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        mean, latent_var, total_var, noise_var, Xt = self._posterior_mean_variance_outputs(X)
        var = self._select_variance_outputs(latent_var, total_var, noise_var)
        thresholds = self._thresholds_like(mean)
        tau = self.tau.to(device=mean.device, dtype=mean.dtype).clamp_min(self.eps)
        boundary_weight = torch.exp(-0.5 * ((mean - thresholds) / tau).pow(2))
        score_outputs = var * boundary_weight
        score_outputs = self._apply_noise_to_output_score(score_outputs, noise_var)
        score = score_outputs.prod(dim=-1) if self.joint_boundary else self._reduce_outputs(score_outputs)
        return self._finalize_pointwise_score(score, X, Xt, name="qHeteroMultiOutputRegressionBoundaryVariance")


class qHeteroMultiOutputRegressionProbabilityOfExceedance(_HeteroMultiOutputRegressionLevelSetBase):
    """Noise-aware probability-of-exceedance / feasibility acquisition."""

    def __init__(
        self,
        model: Model,
        *,
        mode: ProbabilityMode = "above",
        lower: Optional[Sequence[float] | Tensor] = None,
        upper: Optional[Sequence[float] | Tensor] = None,
        temperature: Optional[float | Tensor] = None,
        joint: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, **kwargs)
        if mode not in ("above", "below", "interval"):
            raise ValueError("mode must be 'above', 'below', or 'interval'.")
        self.mode = mode
        self.lower = None if lower is None else torch.as_tensor(lower).reshape(-1)
        self.upper = None if upper is None else torch.as_tensor(upper).reshape(-1)
        self.temperature = None if temperature is None else torch.as_tensor(temperature)
        self.joint = bool(joint)

    def _bounds_like(self, value: Tensor, which: str) -> Tensor:
        bound = self.lower if which == "lower" else self.upper
        if bound is None:
            raise ValueError(f"{which} must be provided when mode='interval'.")
        m = int(value.shape[-1])
        bound = bound.to(device=value.device, dtype=value.dtype)
        if bound.numel() == 1:
            bound = bound.expand(m)
        elif bound.numel() != m:
            raise ValueError(f"{which} length ({bound.numel()}) does not match output dim ({m}).")
        return bound.view(*((1,) * (value.ndim - 1)), m)

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        mean, latent_var, total_var, noise_var, Xt = self._posterior_mean_variance_outputs(X)
        var = self._select_variance_outputs(latent_var, total_var, noise_var)
        std = var.sqrt().clamp_min(self.eps)
        thresholds = self._thresholds_like(mean)
        if self.temperature is not None:
            temp = self.temperature.to(device=mean.device, dtype=mean.dtype).clamp_min(self.eps)
            if self.mode == "above":
                score_outputs = torch.sigmoid((mean - thresholds) / temp)
            elif self.mode == "below":
                score_outputs = torch.sigmoid((thresholds - mean) / temp)
            else:
                lo = self._bounds_like(mean, "lower")
                hi = self._bounds_like(mean, "upper")
                score_outputs = torch.sigmoid((mean - lo) / temp) * torch.sigmoid((hi - mean) / temp)
        else:
            if self.mode == "above":
                score_outputs = _safe_normal_cdf((mean - thresholds) / std)
            elif self.mode == "below":
                score_outputs = _safe_normal_cdf((thresholds - mean) / std)
            else:
                lo = self._bounds_like(mean, "lower")
                hi = self._bounds_like(mean, "upper")
                score_outputs = _safe_normal_cdf((hi - mean) / std) - _safe_normal_cdf((lo - mean) / std)
        score_outputs = score_outputs.clamp_min(0.0)
        score_outputs = self._apply_noise_to_output_score(score_outputs, noise_var)
        score = score_outputs.prod(dim=-1) if self.joint else self._reduce_outputs(score_outputs)
        return self._finalize_pointwise_score(
            score, X, Xt, name="qHeteroMultiOutputRegressionProbabilityOfExceedance"
        )


__all__ = [
    "HeteroMultiOutputRegressionLevelSetScoreObjective",
    "qHeteroMultiOutputRegressionStraddle",
    "qHeteroMultiOutputRegressionJointStraddle",
    "qHeteroMultiOutputRegressionICU",
    "qHeteroMultiOutputRegressionBoundaryVariance",
    "qHeteroMultiOutputRegressionProbabilityOfExceedance",
]
