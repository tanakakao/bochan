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


try:
    from botorch.acquisition.active_learning import (
        qNegIntegratedPosteriorVariance as _BoTorchQNegIntegratedPosteriorVariance,
    )
except Exception:  # pragma: no cover - depends on BoTorch version
    _BoTorchQNegIntegratedPosteriorVariance = None


class _HeteroMultiOutputRegressionActiveLearningBase(AcquisitionFunction):
    """Noise-aware multi-output regression active-learning base.

    This base follows the classification / ordinal hetero acquisition APIs.

    Core conventions:
        - ``reduction`` is q-batch reduction.
        - ``output_reduction`` is multi-output reduction.
        - ``variance_source`` selects latent / total / noise variance.
        - ``noise_combine`` controls how noise avoidance is applied.
        - ``objective`` / ``n_w`` supports InputPerturbation aggregation.
    """

    def __init__(
        self,
        model: Model,
        *,
        reduction: ReductionType = "mean",
        output_reduction: OutputReductionType = "weighted_mean",
        output_weights: Optional[Tensor | Sequence[float]] = None,
        normalize_output_weights: bool = True,
        variance_source: VarianceSource = "latent",
        noise_penalty: Optional[float] = None,
        noise_penalty_lambda: float = 1.0,
        noise_mode: NoiseWeightMode = "inverse_linear",
        noise_combine: NoiseCombineType = "multiply",
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

        # Backward compatibility: old noise_penalty means subtract lambda * noise.
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
        if self.n_w is not None and self.n_w <= 0:
            raise ValueError("n_w must be positive or None.")

        self.X_pending: Optional[Tensor] = None
        self.X_observed: Optional[Tensor] = None
        self.set_X_pending(X_pending)
        self.set_X_observed(X_observed)

    # ------------------------------------------------------------
    # reference / transform helpers
    # ------------------------------------------------------------
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

    # ------------------------------------------------------------
    # output / shape helpers
    # ------------------------------------------------------------
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
        if value.ndim < 1:
            return value
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

    # ------------------------------------------------------------
    # posterior / noise helpers
    # ------------------------------------------------------------
    def _posterior_mean_variance_outputs(self, X: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Return mean_outputs, latent_var_outputs, total_var_outputs, noise_var_outputs, Xt."""
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

    def _posterior_mean_variances(self, X: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        mean_outputs, latent_var_outputs, total_var_outputs, noise_var_outputs, Xt = (
            self._posterior_mean_variance_outputs(X)
        )
        mean = self._reduce_outputs(mean_outputs)
        latent_var = self._reduce_outputs(latent_var_outputs)
        total_var = self._reduce_outputs(total_var_outputs)
        noise_var = self._reduce_outputs(noise_var_outputs)

        mean = self._align_pointwise_score_to_X(mean, Xt, name="posterior.mean")
        latent_var = self._align_pointwise_score_to_X(latent_var, Xt, name="latent variance")
        total_var = self._align_pointwise_score_to_X(total_var, Xt, name="total variance")
        noise_var = self._align_pointwise_score_to_X(noise_var, Xt, name="noise variance")
        return mean, latent_var, total_var, noise_var, Xt

    def _select_variance(self, latent_var: Tensor, total_var: Tensor, noise_var: Tensor) -> Tensor:
        if self.variance_source == "latent":
            return latent_var
        if self.variance_source == "total":
            return total_var
        if self.variance_source == "noise":
            return noise_var
        raise ValueError(f"Unknown variance_source: {self.variance_source!r}.")

    def _noise_weight(self, noise_var: Tensor) -> Tensor:
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

    def _apply_noise_to_score(self, score: Tensor, noise_var: Tensor) -> Tensor:
        noise_var = self._align_pointwise_score_to_X(noise_var, _ensure_q_batch(noise_var).unsqueeze(-1) if noise_var.ndim == 1 else _ensure_q_batch(noise_var), name="noise variance alignment") if False else noise_var

        if self.noise_combine == "none":
            return score
        if self.noise_combine == "subtract":
            return score - self.noise_penalty_lambda * noise_var
        if self.noise_combine == "multiply":
            weight = self._noise_weight(noise_var)
            if weight.shape != score.shape:
                if weight.numel() == score.numel():
                    weight = weight.reshape_as(score)
                else:
                    weight = self._reduce_outputs(weight)
                    if weight.shape != score.shape and weight.numel() == score.numel():
                        weight = weight.reshape_as(score)
            return score * weight
        raise ValueError(f"Unknown noise_combine={self.noise_combine!r}.")

    # ------------------------------------------------------------
    # penalties / objective / reductions
    # ------------------------------------------------------------
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
                Xt,
                self.X_pending,
                weight=self.pending_penalty_weight,
                beta=self.pending_penalty_beta,
            )
            + self._reference_penalty_per_point(
                Xt,
                self.X_observed,
                weight=self.observed_penalty_weight,
                beta=self.observed_penalty_beta,
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


class qHeteroMultiOutputRegressionPosteriorVariance(_HeteroMultiOutputRegressionActiveLearningBase):
    """Noise-aware posterior variance acquisition for multi-output regression."""

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        _, latent_var, total_var, noise_var, Xt = self._posterior_mean_variances(X)
        score = self._select_variance(latent_var, total_var, noise_var)
        score = self._apply_noise_to_score(score, noise_var)
        return self._finalize_pointwise_score(score, X, Xt, name="qHeteroMultiOutputRegressionPosteriorVariance")


class qHeteroMultiOutputRegressionPredictiveEntropy(_HeteroMultiOutputRegressionActiveLearningBase):
    """Noise-aware predictive entropy acquisition for multi-output regression."""

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        _, latent_var, total_var, noise_var, Xt = self._posterior_mean_variances(X)
        var = total_var if self.variance_source == "total" else self._select_variance(latent_var, total_var, noise_var)
        entropy = 0.5 * torch.log(
            torch.as_tensor(2.0 * torch.pi * torch.e, device=var.device, dtype=var.dtype)
            * var.clamp_min(self.eps)
        )
        score = self._apply_noise_to_score(entropy, noise_var)
        return self._finalize_pointwise_score(score, X, Xt, name="qHeteroMultiOutputRegressionPredictiveEntropy")


class qHeteroMultiOutputRegressionBALD(_HeteroMultiOutputRegressionActiveLearningBase):
    """Noise-aware BALD / mutual-information acquisition for multi-output regression."""

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        _, _, total_var, noise_var, Xt = self._posterior_mean_variances(X)
        score = 0.5 * torch.log(total_var.clamp_min(self.eps) / noise_var.clamp_min(self.eps))
        score = self._apply_noise_to_score(score, noise_var)
        return self._finalize_pointwise_score(score, X, Xt, name="qHeteroMultiOutputRegressionBALD")


class qHeteroMultiOutputRegressionIntegratedPosteriorVarianceProxy(
    _HeteroMultiOutputRegressionActiveLearningBase
):
    """Lightweight heteroscedastic integrated posterior variance proxy.

    This does not fantasize.  It scores candidates by coverage of high
    variance reference regions and applies noise-aware weighting / penalty.
    """

    def __init__(
        self,
        model: Model,
        X_ref: Tensor,
        *,
        kernel_lengthscale: float = 0.2,
        normalize_weights: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, **kwargs)
        if X_ref.ndim != 2:
            raise ValueError(f"X_ref must have shape [n_ref, d]. Got {tuple(X_ref.shape)}.")
        self.register_buffer("X_ref", X_ref.detach().clone())
        self.kernel_lengthscale = float(kernel_lengthscale)
        self.normalize_weights = bool(normalize_weights)

    def _reference_score(self) -> Tensor:
        _, latent_var, total_var, noise_var, _ = self._posterior_mean_variances(self.X_ref)
        n_ref = int(self.X_ref.shape[-2])
        score = self._select_variance(latent_var, total_var, noise_var)
        score = self._apply_noise_to_score(score, noise_var)
        score = self._aggregate_n_w_if_needed(
            score,
            q=n_ref,
            context="qHeteroMultiOutputRegressionIntegratedPosteriorVarianceProxy reference score",
        )
        if score.shape[-1] != n_ref:
            raise RuntimeError(
                "Reference score must have last dimension n_ref. "
                f"score.shape={tuple(score.shape)}, n_ref={n_ref}."
            )
        return score

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        raw_X = _ensure_q_batch(X)
        Xt = self._apply_input_transform_for_distance(raw_X)

        ref_score = self._reference_score()
        X_ref_t = self._reference_to_distance_space(self.X_ref, like=Xt)
        if X_ref_t is None:
            raise RuntimeError("X_ref unexpectedly became None after transform.")
        X_ref_2d = X_ref_t.reshape(-1, X_ref_t.shape[-1])

        if ref_score.ndim > 1:
            while ref_score.ndim > 1:
                ref_score = ref_score.mean(dim=0)

        if ref_score.shape[-1] != X_ref_2d.shape[-2]:
            n_ref = int(self.X_ref.shape[-2])
            if X_ref_2d.shape[-2] % n_ref == 0:
                n_w_ref = X_ref_2d.shape[-2] // n_ref
                X_ref_2d = X_ref_2d.reshape(n_ref, n_w_ref, X_ref_2d.shape[-1]).mean(dim=1)
            if ref_score.shape[-1] != X_ref_2d.shape[-2]:
                raise RuntimeError(
                    "Reference score / reference point mismatch. "
                    f"ref_score.shape={tuple(ref_score.shape)}, X_ref_2d.shape={tuple(X_ref_2d.shape)}."
                )

        d2 = torch.cdist(Xt.reshape(-1, Xt.shape[-1]), X_ref_2d).pow(2)
        d2 = d2.reshape(*Xt.shape[:-1], X_ref_2d.shape[-2])

        ls2 = max(self.kernel_lengthscale ** 2, self.eps)
        weights = torch.exp(-0.5 * d2 / ls2)
        if self.normalize_weights:
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(self.eps)

        view_shape = (1,) * (weights.ndim - 1) + (ref_score.shape[-1],)
        score = (weights * ref_score.view(*view_shape)).sum(dim=-1)

        return self._finalize_pointwise_score(
            score,
            raw_X,
            Xt,
            name="qHeteroMultiOutputRegressionIntegratedPosteriorVarianceProxy",
        )


__all__ = [
    "qHeteroMultiOutputRegressionPredictiveEntropy",
    "qHeteroMultiOutputRegressionBALD",
    "qHeteroMultiOutputRegressionPosteriorVariance",
    "qHeteroMultiOutputRegressionIntegratedPosteriorVarianceProxy",
]
