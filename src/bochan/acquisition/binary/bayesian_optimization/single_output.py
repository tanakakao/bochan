from __future__ import annotations

import math
from typing import Callable, Literal, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from bochan.acquisition.binary._likelihood import latent_samples_to_binary_probabilities

from botorch.acquisition import AcquisitionFunction
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.models import ModelListGP
from botorch.models.gpytorch import ModelListGPyTorchModel
from botorch.models.model import Model
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.transforms import match_batch_shape, t_batch_mode_transform

from bochan.acquisition.binary.base import ReductionType, _BinaryClassificationAcqBase

from ._utils import (
    apply_score_objective,
    compute_binary_best_f,
    ensure_q_batch,
    reshape_binary_samples,
    to_probability,
)


PoFMode = Literal["mc_likelihood", "mc_sigmoid", "latent_cdf"]
QFeasMode = Literal["prod", "mean", "min", "max"]
QBatchMode = Literal["pointwise", "joint"]
CombineMode = Literal["product", "log_product", "penalty"]
BaseTransformMode = Literal["identity", "clamp_nonnegative", "softplus"]


def _finalize_binary_acq_output_to_batch(
    value: Tensor,
    X: Tensor,
    *,
    name: str,
) -> Tensor:
    """binary BO acquisition outputをBoTorchのt-batch shapeに揃える。"""
    Xq = ensure_q_batch(X)
    target = tuple(Xq.shape[:-2])
    out = value

    if out.shape == target:
        return out
    if len(target) == 0:
        return out if out.ndim == 0 else out.mean()
    if out.ndim == 0:
        return out.expand(*target)

    while out.ndim > len(target):
        out = out.mean(dim=0)
        if out.shape == target:
            return out

    if out.shape == target:
        return out
    if out.numel() == int(torch.tensor(target).prod().item()):
        return out.reshape(target)
    if out.ndim == 1 and len(target) == 1:
        if out.shape[0] == target[0]:
            return out
        return out.mean().expand(*target)

    raise RuntimeError(
        f"{name}: could not align acquisition output to t-batch shape. "
        f"value.shape={tuple(value.shape)}, target={target}, X.shape={tuple(Xq.shape)}."
    )


def _coerce_reference_tensor(
    X_ref: Tensor | list[Tensor] | tuple[Tensor, ...] | None,
    *,
    ref: Tensor | None = None,
) -> Tensor | None:
    """参照点を単一Tensorへ揃える。"""
    if X_ref is None:
        return None
    if torch.is_tensor(X_ref):
        out = X_ref
    elif isinstance(X_ref, (list, tuple)):
        tensors = [
            _coerce_reference_tensor(item, ref=ref)
            for item in X_ref
            if item is not None
        ]
        tensors = [
            item
            for item in tensors
            if item is not None and item.numel() > 0
        ]
        if len(tensors) == 0:
            return None
        out = torch.cat(
            [item.reshape(-1, item.shape[-1]) for item in tensors],
            dim=-2,
        )
    else:
        raise TypeError(
            "Reference points must be Tensor, sequence of Tensors, or None. "
            f"Got {type(X_ref)}."
        )
    if ref is not None:
        out = out.to(device=ref.device, dtype=ref.dtype)
    return out.detach()


def _resolve_observed_X(model: Model) -> Tensor | None:
    """モデルから学習入力をbest-effortで取り出す。"""
    for attr in ("train_X_original", "train_X", "train_inputs_raw"):
        value = getattr(model, attr, None)
        if value is not None:
            return value[0] if isinstance(value, tuple) else value

    value = getattr(model, "train_inputs", None)
    if isinstance(value, tuple) and len(value) > 0:
        return value[0]

    inner = getattr(model, "model", None)
    if inner is not None and inner is not model:
        value = getattr(inner, "train_inputs", None)
        if isinstance(value, tuple) and len(value) > 0:
            return value[0]
    return None


class qBinaryProbabilityOfFeasibility(_BinaryClassificationAcqBase):
    """binary classification用probability of feasibility acquisition。"""

    def __init__(
        self,
        model,
        num_samples: int = 32,
        threshold: float = 0.0,
        mode: PoFMode = "mc_likelihood",
        reduction: ReductionType = "mean",
        pending_penalty_weight: float = 0.0,
        pending_penalty_beta: float = 10.0,
        eps: float = 1e-6,
        objective: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
    ):
        try:
            super().__init__(
                model=model,
                reduction=reduction,
                pending_penalty_weight=pending_penalty_weight,
                pending_penalty_beta=pending_penalty_beta,
                eps=eps,
                objective=objective,
            )
        except TypeError:
            super().__init__(
                model=model,
                reduction=reduction,
                pending_penalty_weight=pending_penalty_weight,
                pending_penalty_beta=pending_penalty_beta,
                eps=eps,
            )
        self.num_samples = int(num_samples)
        self.threshold = float(threshold)
        self.mode = mode
        self.objective = objective

    def _mc_likelihood_prob(self, latent_dist, orig: torch.Size) -> Tensor:
        f_samples = latent_dist.rsample(torch.Size([self.num_samples]))
        expected = self.num_samples * math.prod(orig)
        if f_samples.numel() != expected:
            raise RuntimeError(
                f"Unexpected sample shape: got {tuple(f_samples.shape)}, "
                f"numel={f_samples.numel()}, expected={expected}"
            )
        f_samples = f_samples.reshape(self.num_samples, *orig)
        return latent_samples_to_binary_probabilities(self.model, f_samples, eps=self.eps, name="f_samples via binary likelihood").clamp(
            self.eps,
            1.0 - self.eps,
        ).mean(dim=0)

    def _latent_cdf_prob(self, latent_dist, orig: torch.Size) -> Tensor:
        mu = self._reshape_pointwise_tensor(latent_dist.mean, orig)
        var = self._reshape_pointwise_tensor(
            latent_dist.variance,
            orig,
        ).clamp_min(self.eps)
        sigma = var.sqrt()
        z = (mu - self.threshold) / sigma
        normal = torch.distributions.Normal(
            torch.zeros_like(z),
            torch.ones_like(z),
        )
        return normal.cdf(z).clamp(self.eps, 1.0 - self.eps)

    def _pointwise_pof_from_latent_dist(
        self,
        latent_dist,
        orig: torch.Size,
    ) -> Tensor:
        if self.mode in {"mc_likelihood", "mc_sigmoid"}:
            return self._mc_likelihood_prob(latent_dist, orig)
        if self.mode == "latent_cdf":
            return self._latent_cdf_prob(latent_dist, orig)
        raise ValueError(f"Unknown mode: {self.mode}")

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._prepare_eval()
        X_in = X if X.ndim > 2 else X.unsqueeze(0)
        original_batch_shape = X_in.shape[:-2]

        latent_dist, orig, Xt = self._get_latent_dist_and_orig(X)
        score = self._pointwise_pof_from_latent_dist(latent_dist, orig)

        penalty = self._pending_penalty_per_point(Xt)
        if penalty.shape == score.shape:
            score = score - penalty
        elif penalty.numel() == score.numel():
            score = score - penalty.reshape_as(score)
        elif self.pending_penalty_weight > 0:
            raise RuntimeError(
                "Pending penalty shape mismatch: "
                f"score={tuple(score.shape)}, penalty={tuple(penalty.shape)}"
            )

        score = apply_score_objective(
            self,
            score,
            X=X,
            attr_name="objective",
            name="PoF",
        )
        out = self._reduce_q(score)
        self._check_output_shape(out, original_batch_shape, "PoF")
        return out


class _BinaryProbabilityBOBase(MCAcquisitionFunction):
    """Binary probability-space BO acquisition共通基底。

    ``q_mode="joint"``はBoTorchのqEI/qPI/qUCBと同様に、joint posteriorを
    sampleし、sampleごとにq方向を最大化する。``q_mode="pointwise"``は
    各候補点のscoreを個別に計算してq方向へ集約する。
    """

    def __init__(
        self,
        model: Model,
        *,
        sampler: Optional[SobolQMCNormalSampler] = None,
        apply_sigmoid_if_needed: bool = True,
        q_mode: QBatchMode = "pointwise",
        reduction: ReductionType = "mean",
        X_pending: Tensor | None = None,
        X_observed: Tensor | None = None,
        X_baseline: Tensor | None = None,
        pending_penalty_weight: float = 0.0,
        pending_penalty_beta: float = 10.0,
        observed_penalty_weight: float = 0.0,
        observed_penalty_beta: float = 10.0,
        same_batch_penalty_weight: float = 0.0,
        same_batch_penalty_beta: float = 10.0,
        eps: float = 1e-6,
        objective: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
        **kwargs,
    ) -> None:
        if sampler is None:
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))
        if q_mode not in {"pointwise", "joint"}:
            raise ValueError("q_mode must be 'pointwise' or 'joint'.")
        if reduction not in {"mean", "sum", "max"}:
            raise ValueError("reduction must be 'mean', 'sum', or 'max'.")

        super().__init__(model=model, sampler=sampler, objective=None, **kwargs)

        self.apply_sigmoid_if_needed = bool(apply_sigmoid_if_needed)
        self.q_mode = q_mode
        self.reduction = reduction
        self.pending_penalty_weight = float(pending_penalty_weight)
        self.pending_penalty_beta = float(pending_penalty_beta)
        self.observed_penalty_weight = float(observed_penalty_weight)
        self.observed_penalty_beta = float(observed_penalty_beta)
        self.same_batch_penalty_weight = float(same_batch_penalty_weight)
        self.same_batch_penalty_beta = float(same_batch_penalty_beta)
        self.eps = float(eps)
        self.score_objective = objective

        observed = X_observed
        if observed is None:
            observed = X_baseline
        if observed is None:
            observed = _resolve_observed_X(model)
        self.X_observed = _coerce_reference_tensor(observed)
        self.set_X_pending(X_pending)

    def set_X_pending(self, X_pending: Tensor | None = None) -> None:
        self.X_pending = _coerce_reference_tensor(X_pending)

    @property
    def _sample_ndim(self) -> int:
        return len(getattr(self.sampler, "sample_shape", torch.Size([1])))

    def _mean_over_sample_dims(self, value: Tensor) -> Tensor:
        if self._sample_ndim <= 0:
            return value
        return value.mean(dim=tuple(range(self._sample_ndim)))

    def _apply_input_transform(self, X: Tensor) -> Tensor:
        X = ensure_q_batch(X)

        for name in ("_to_internal", "_to_latent", "_to_training_feature_space"):
            transform = getattr(self.model, name, None)
            if callable(transform):
                Xt = transform(X)
                if isinstance(Xt, tuple):
                    Xt = Xt[0]
                return ensure_q_batch(Xt)

        input_transform = getattr(self.model, "input_transform", None)
        if callable(input_transform):
            Xt = input_transform(X)
            if isinstance(Xt, tuple):
                Xt = Xt[0]
            return ensure_q_batch(Xt)
        return X

    def _reference_to_transformed(
        self,
        X_ref: Tensor | list[Tensor] | tuple[Tensor, ...] | None,
        *,
        ref: Tensor,
    ) -> Tensor | None:
        X_ref = _coerce_reference_tensor(X_ref, ref=ref)
        if X_ref is None or X_ref.numel() == 0:
            return None
        Xt = self._apply_input_transform(X_ref)
        return Xt.reshape(-1, Xt.shape[-1]).to(ref)

    def _reference_penalty_per_point(
        self,
        Xt: Tensor,
        X_ref: Tensor | list[Tensor] | tuple[Tensor, ...] | None,
        *,
        weight: float,
        beta: float,
    ) -> Tensor:
        Xt = ensure_q_batch(Xt)
        if weight <= 0.0:
            return Xt.new_zeros(Xt.shape[:-1])

        ref = self._reference_to_transformed(X_ref, ref=Xt)
        if ref is None:
            return Xt.new_zeros(Xt.shape[:-1])

        distance = torch.cdist(
            Xt.reshape(-1, Xt.shape[-1]),
            ref,
        ).min(dim=-1).values
        return weight * torch.exp(
            -beta * distance.reshape(*Xt.shape[:-1])
        )

    def _pending_penalty_per_point(self, Xt: Tensor) -> Tensor:
        return self._reference_penalty_per_point(
            Xt,
            self.X_pending,
            weight=self.pending_penalty_weight,
            beta=self.pending_penalty_beta,
        )

    def _observed_penalty_per_point(self, Xt: Tensor) -> Tensor:
        return self._reference_penalty_per_point(
            Xt,
            self.X_observed,
            weight=self.observed_penalty_weight,
            beta=self.observed_penalty_beta,
        )

    def _same_batch_penalty(self, Xt: Tensor) -> Tensor:
        Xt = ensure_q_batch(Xt)
        if self.same_batch_penalty_weight <= 0.0 or Xt.shape[-2] <= 1:
            return Xt.new_zeros(Xt.shape[:-2])

        batch_shape = Xt.shape[:-2]
        q = int(Xt.shape[-2])
        Xb = Xt.reshape(-1, q, Xt.shape[-1])
        distance = torch.cdist(Xb, Xb)
        eye = torch.eye(
            q,
            device=Xt.device,
            dtype=torch.bool,
        ).unsqueeze(0)
        distance = distance.masked_fill(eye, float("inf"))
        penalty = (
            0.5
            * self.same_batch_penalty_weight
            * torch.exp(-self.same_batch_penalty_beta * distance).sum(
                dim=(-1, -2)
            )
        )
        return penalty.reshape(*batch_shape)

    def _reduce_q(self, score: Tensor) -> Tensor:
        if self.reduction == "mean":
            return score.mean(dim=-1)
        if self.reduction == "sum":
            return score.sum(dim=-1)
        if self.reduction == "max":
            return score.max(dim=-1).values
        raise ValueError(f"Unknown reduction: {self.reduction!r}.")

    def _pointwise_score_to_value(
        self,
        score: Tensor,
        X: Tensor,
    ) -> Tensor:
        raw_X = ensure_q_batch(X)
        Xt = self._apply_input_transform(raw_X)

        score = score - self._pending_penalty_per_point(Xt)
        score = score - self._observed_penalty_per_point(Xt)
        value = self._reduce_q(score)
        value = value - self._same_batch_penalty(Xt)
        return _finalize_binary_acq_output_to_batch(
            value,
            raw_X,
            name=self.__class__.__name__,
        )

    def _joint_X(self, X: Tensor) -> Tensor:
        X = ensure_q_batch(X)
        if self.X_pending is None:
            return X
        X_pending = self.X_pending.to(device=X.device, dtype=X.dtype)
        return torch.cat(
            [X, match_batch_shape(X_pending, X)],
            dim=-2,
        )

    @staticmethod
    def _squeeze_binary_output_dim_if_present(
        probs: Tensor,
        X: Tensor,
    ) -> Tensor:
        if probs.ndim > X.ndim and probs.shape[-1] == 1:
            return probs.squeeze(-1)
        return probs

    def _posterior_samples_as_prob(self, X: Tensor) -> Tensor:
        """posterior samplesをbinary probability samplesへ変換する。"""
        X = ensure_q_batch(X)

        if self.apply_sigmoid_if_needed and hasattr(
            self.model,
            "latent_posterior",
        ):
            post = self.model.latent_posterior(X)
            samples = self.get_posterior_samples(post)
            probs = latent_samples_to_binary_probabilities(self.model, samples, eps=self.eps, name="samples via binary likelihood").clamp(
                self.eps,
                1.0 - self.eps,
            )
        else:
            post = self.model.posterior(X)
            samples = self.get_posterior_samples(post)
            probs = to_probability(samples, apply_sigmoid_if_needed=self.apply_sigmoid_if_needed, eps=self.eps, name='posterior samples', model=self.model)

        if self.score_objective is not None:
            probs = self.score_objective(probs, X=X)

        probs = self._squeeze_binary_output_dim_if_present(probs, X)
        return reshape_binary_samples(probs, X)

    def _resolve_best_f(
        self,
        best_f: float | Tensor | None,
        *,
        best_f_margin: float,
        best_f_quantile: float | None,
    ) -> Tensor:
        if best_f is not None:
            return torch.as_tensor(best_f)

        train_X = _resolve_observed_X(self.model)
        if train_X is None:
            raise ValueError(
                "best_f was not provided and training inputs could not be "
                "resolved from the model."
            )

        objective = self.score_objective
        risk_type = getattr(objective, "risk_type", None)
        alpha = float(getattr(objective, "alpha", 0.5))
        return compute_binary_best_f(
            self.model,
            train_X,
            apply_sigmoid_if_needed=True,
            risk_type=risk_type,
            alpha=alpha,
            eps=self.eps,
            best_f_margin=best_f_margin,
            best_f_quantile=best_f_quantile,
        )


class qBinaryExpectedImprovement(_BinaryProbabilityBOBase):
    """Positive-class probabilityに対するExpected Improvement。

    ``q_mode="joint"``はBoTorch qEIと同じjoint max、既定の
    ``q_mode="pointwise"``は各点のEIをq方向へ集約する。
    """

    def __init__(
        self,
        model: Model,
        best_f: float | Tensor | None = None,
        *,
        sampler: Optional[SobolQMCNormalSampler] = None,
        apply_sigmoid_if_needed: bool = True,
        q_mode: QBatchMode = "pointwise",
        reduction: ReductionType = "mean",
        X_pending: Tensor | None = None,
        X_observed: Tensor | None = None,
        X_baseline: Tensor | None = None,
        pending_penalty_weight: float = 0.0,
        pending_penalty_beta: float = 10.0,
        observed_penalty_weight: float = 0.0,
        observed_penalty_beta: float = 10.0,
        same_batch_penalty_weight: float = 0.0,
        same_batch_penalty_beta: float = 10.0,
        best_f_margin: float = 1e-4,
        best_f_quantile: float | None = None,
        eps: float = 1e-6,
        objective: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
    ) -> None:
        super().__init__(
            model=model,
            sampler=sampler,
            apply_sigmoid_if_needed=apply_sigmoid_if_needed,
            q_mode=q_mode,
            reduction=reduction,
            X_pending=X_pending,
            X_observed=X_observed,
            X_baseline=X_baseline,
            pending_penalty_weight=pending_penalty_weight,
            pending_penalty_beta=pending_penalty_beta,
            observed_penalty_weight=observed_penalty_weight,
            observed_penalty_beta=observed_penalty_beta,
            same_batch_penalty_weight=same_batch_penalty_weight,
            same_batch_penalty_beta=same_batch_penalty_beta,
            eps=eps,
            objective=objective,
        )
        resolved_best_f = self._resolve_best_f(
            best_f,
            best_f_margin=best_f_margin,
            best_f_quantile=best_f_quantile,
        )
        self.register_buffer("best_f", resolved_best_f)

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        raw_X = ensure_q_batch(X)
        eval_X = self._joint_X(raw_X) if self.q_mode == "joint" else raw_X
        probs = self._posterior_samples_as_prob(eval_X)
        improvement = (
            probs - self.best_f.to(probs)
        ).clamp_min(0.0)

        if self.q_mode == "joint":
            value = self._mean_over_sample_dims(
                improvement.max(dim=-1).values
            )
            value = value - self._same_batch_penalty(
                self._apply_input_transform(raw_X)
            )
            return _finalize_binary_acq_output_to_batch(
                value,
                raw_X,
                name=self.__class__.__name__,
            )

        pointwise = self._mean_over_sample_dims(improvement)
        return self._pointwise_score_to_value(pointwise, raw_X)


class qBinaryProbabilityOfImprovement(_BinaryProbabilityBOBase):
    """Positive-class probabilityに対するProbability of Improvement。"""

    def __init__(
        self,
        model: Model,
        best_f: float | Tensor | None = None,
        *,
        tau: float = 1e-3,
        sampler: Optional[SobolQMCNormalSampler] = None,
        apply_sigmoid_if_needed: bool = True,
        q_mode: QBatchMode = "pointwise",
        reduction: ReductionType = "mean",
        X_pending: Tensor | None = None,
        X_observed: Tensor | None = None,
        X_baseline: Tensor | None = None,
        pending_penalty_weight: float = 0.0,
        pending_penalty_beta: float = 10.0,
        observed_penalty_weight: float = 0.0,
        observed_penalty_beta: float = 10.0,
        same_batch_penalty_weight: float = 0.0,
        same_batch_penalty_beta: float = 10.0,
        best_f_margin: float = 1e-4,
        best_f_quantile: float | None = None,
        eps: float = 1e-6,
        objective: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
    ) -> None:
        super().__init__(
            model=model,
            sampler=sampler,
            apply_sigmoid_if_needed=apply_sigmoid_if_needed,
            q_mode=q_mode,
            reduction=reduction,
            X_pending=X_pending,
            X_observed=X_observed,
            X_baseline=X_baseline,
            pending_penalty_weight=pending_penalty_weight,
            pending_penalty_beta=pending_penalty_beta,
            observed_penalty_weight=observed_penalty_weight,
            observed_penalty_beta=observed_penalty_beta,
            same_batch_penalty_weight=same_batch_penalty_weight,
            same_batch_penalty_beta=same_batch_penalty_beta,
            eps=eps,
            objective=objective,
        )
        resolved_best_f = self._resolve_best_f(
            best_f,
            best_f_margin=best_f_margin,
            best_f_quantile=best_f_quantile,
        )
        self.register_buffer("best_f", resolved_best_f)
        self.register_buffer("tau", torch.as_tensor(tau))

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        raw_X = ensure_q_batch(X)
        eval_X = self._joint_X(raw_X) if self.q_mode == "joint" else raw_X
        probs = self._posterior_samples_as_prob(eval_X)
        tau = self.tau.to(probs).clamp_min(self.eps)
        indicator = torch.sigmoid(
            (probs - self.best_f.to(probs)) / tau
        )

        if self.q_mode == "joint":
            value = self._mean_over_sample_dims(
                indicator.max(dim=-1).values
            )
            value = value - self._same_batch_penalty(
                self._apply_input_transform(raw_X)
            )
            return _finalize_binary_acq_output_to_batch(
                value,
                raw_X,
                name=self.__class__.__name__,
            )

        pointwise = self._mean_over_sample_dims(indicator)
        return self._pointwise_score_to_value(pointwise, raw_X)


class qBinaryUpperConfidenceBound(_BinaryProbabilityBOBase):
    """Positive-class probabilityに対するUpper Confidence Bound。"""

    def __init__(
        self,
        model: Model,
        beta: float | Tensor = 2.0,
        *,
        sampler: Optional[SobolQMCNormalSampler] = None,
        apply_sigmoid_if_needed: bool = True,
        q_mode: QBatchMode = "pointwise",
        reduction: ReductionType = "mean",
        X_pending: Tensor | None = None,
        X_observed: Tensor | None = None,
        X_baseline: Tensor | None = None,
        pending_penalty_weight: float = 0.0,
        pending_penalty_beta: float = 10.0,
        observed_penalty_weight: float = 0.0,
        observed_penalty_beta: float = 10.0,
        same_batch_penalty_weight: float = 0.0,
        same_batch_penalty_beta: float = 10.0,
        eps: float = 1e-6,
        objective: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
    ) -> None:
        super().__init__(
            model=model,
            sampler=sampler,
            apply_sigmoid_if_needed=apply_sigmoid_if_needed,
            q_mode=q_mode,
            reduction=reduction,
            X_pending=X_pending,
            X_observed=X_observed,
            X_baseline=X_baseline,
            pending_penalty_weight=pending_penalty_weight,
            pending_penalty_beta=pending_penalty_beta,
            observed_penalty_weight=observed_penalty_weight,
            observed_penalty_beta=observed_penalty_beta,
            same_batch_penalty_weight=same_batch_penalty_weight,
            same_batch_penalty_beta=same_batch_penalty_beta,
            eps=eps,
            objective=objective,
        )
        self.register_buffer("beta", torch.as_tensor(beta))

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        raw_X = ensure_q_batch(X)
        eval_X = self._joint_X(raw_X) if self.q_mode == "joint" else raw_X
        probs = self._posterior_samples_as_prob(eval_X)

        sample_dims = tuple(range(self._sample_ndim))
        mean = probs.mean(dim=sample_dims, keepdim=True)
        beta_prime = torch.sqrt(
            self.beta.to(probs) * probs.new_tensor(math.pi / 2.0)
        )
        sample_ucb = mean + beta_prime * (probs - mean).abs()

        if self.q_mode == "joint":
            value = self._mean_over_sample_dims(
                sample_ucb.max(dim=-1).values
            )
            value = value - self._same_batch_penalty(
                self._apply_input_transform(raw_X)
            )
            return _finalize_binary_acq_output_to_batch(
                value,
                raw_X,
                name=self.__class__.__name__,
            )

        pointwise = self._mean_over_sample_dims(sample_ucb)
        return self._pointwise_score_to_value(pointwise, raw_X)


class _qBinaryFeasibilityWeightedAcquisition(AcquisitionFunction):
    """Feasibility-weighted wrapper for arbitrary objective acquisition。"""

    def __init__(
        self,
        objective_acqf: AcquisitionFunction,
        feasibility_model,
        num_pof_samples: int = 32,
        threshold: float = 0.0,
        pof_mode: PoFMode = "mc_likelihood",
        combine_mode: CombineMode = "product",
        q_feas_mode: QFeasMode = "prod",
        feasibility_power: float = 1.0,
        base_transform: BaseTransformMode = "identity",
        penalty_weight: float = 1.0,
        eps: float = 1e-8,
        feasibility_objective: Optional[
            Callable[[Tensor, Optional[Tensor]], Tensor]
        ] = None,
    ):
        if isinstance(
            feasibility_model,
            (ModelListGP, ModelListGPyTorchModel),
        ):
            feasibility_model = feasibility_model.models[0]

        super().__init__(
            getattr(objective_acqf, "model", feasibility_model)
        )
        self.objective_acqf = objective_acqf
        self.feasibility_model = feasibility_model
        self.num_pof_samples = int(num_pof_samples)
        self.threshold = float(threshold)
        self.pof_mode = pof_mode
        self.combine_mode = combine_mode
        self.q_feas_mode = q_feas_mode
        self.feasibility_power = float(feasibility_power)
        self.base_transform = base_transform
        self.penalty_weight = float(penalty_weight)
        self.eps = float(eps)
        self.feasibility_objective = feasibility_objective
        self.set_X_pending(None)

    def set_X_pending(
        self,
        X_pending: Tensor | None = None,
    ) -> None:
        self.X_pending = X_pending
        if hasattr(self.objective_acqf, "set_X_pending"):
            self.objective_acqf.set_X_pending(X_pending)

    def _pof_acqf(self) -> qBinaryProbabilityOfFeasibility:
        acqf = qBinaryProbabilityOfFeasibility(
            model=self.feasibility_model,
            num_samples=self.num_pof_samples,
            threshold=self.threshold,
            mode=self.pof_mode,
            reduction="mean",
            eps=self.eps,
            objective=self.feasibility_objective,
        )
        if getattr(self, "X_pending", None) is not None:
            acqf.set_X_pending(self.X_pending)
        return acqf

    def _transform_objective(self, base_val: Tensor) -> Tensor:
        if self.base_transform == "identity":
            return base_val
        if self.base_transform == "clamp_nonnegative":
            return base_val.clamp_min(0.0)
        if self.base_transform == "softplus":
            return F.softplus(base_val)
        raise ValueError(
            f"Unknown base_transform: {self.base_transform}"
        )

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        base_val = self.objective_acqf(X)

        pof_acqf = self._pof_acqf()
        latent_dist, orig, _ = pof_acqf._get_latent_dist_and_orig(X)
        pof_point = pof_acqf._pointwise_pof_from_latent_dist(
            latent_dist,
            orig,
        )
        pof_point = apply_score_objective(
            self,
            pof_point,
            X=X,
            attr_name="feasibility_objective",
            name="FeasibilityWeightedAcquisitionBinary",
        )

        if self.q_feas_mode == "prod":
            q_pof = pof_point.prod(dim=-1)
        elif self.q_feas_mode == "mean":
            q_pof = pof_point.mean(dim=-1)
        elif self.q_feas_mode == "min":
            q_pof = pof_point.min(dim=-1).values
        elif self.q_feas_mode == "max":
            q_pof = pof_point.max(dim=-1).values
        else:
            raise ValueError(
                f"Unknown q_feas_mode: {self.q_feas_mode}"
            )

        q_pof = q_pof.clamp(self.eps, 1.0 - self.eps)

        if self.combine_mode == "product":
            value = self._transform_objective(
                base_val
            ) * q_pof.pow(self.feasibility_power)
        elif self.combine_mode == "log_product":
            value = base_val + self.feasibility_power * torch.log(q_pof)
        elif self.combine_mode == "penalty":
            value = (
                self._transform_objective(base_val)
                - self.penalty_weight * (1.0 - q_pof)
            )
        else:
            raise ValueError(
                f"Unknown combine_mode: {self.combine_mode}"
            )

        return _finalize_binary_acq_output_to_batch(
            value,
            X,
            name="qBinaryFeasibilityWeightedAcquisition",
        )


__all__ = [
    "QBatchMode",
    "qBinaryProbabilityOfFeasibility",
    "qBinaryExpectedImprovement",
    "qBinaryProbabilityOfImprovement",
    "qBinaryUpperConfidenceBound",
]
