from __future__ import annotations

import math
from typing import Callable, Literal, Optional

import torch
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.models.model import Model
from botorch.sampling.base import MCSampler
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.transforms import match_batch_shape, t_batch_mode_transform
from torch import Tensor

from .single_output import (
    _apply_input_transform_for_pending,
    _cat_dims_from_model,
    _coerce_pending_to_tensor,
    _mean_over_sample_dims,
    _normalize_utility_samples,
    _pairwise_distance2,
    _resolve_observed_X,
    ensure_q_batch,
)


OrdinalQBatchMode = Literal["pointwise", "joint"]
OrdinalQReduction = Literal["mean", "sum", "max"]


def _finalize_output(value: Tensor, X: Tensor, *, name: str) -> Tensor:
    target = torch.Size(ensure_q_batch(X).shape[:-2])
    if value.shape == target:
        return value
    if len(target) == 0:
        return value if value.ndim == 0 else value.mean()
    if value.ndim == 0:
        return value.expand(target)
    while value.ndim > len(target):
        value = value.mean(dim=0)
        if value.shape == target:
            return value
    if value.numel() == math.prod(target):
        return value.reshape(target)
    raise RuntimeError(
        f"{name}: output shape mismatch. value.shape={tuple(value.shape)}, "
        f"expected={tuple(target)}."
    )


class _OrdinalPointwiseUtilityBOBase(MCAcquisitionFunction):
    """Common ordinal utility BO base with pointwise and joint q semantics."""

    def __init__(
        self,
        model: Model,
        *,
        objective: Callable[[Tensor, Optional[Tensor]], Tensor],
        sampler: Optional[MCSampler] = None,
        q_mode: OrdinalQBatchMode = "pointwise",
        reduction: OrdinalQReduction = "mean",
        X_pending: Optional[Tensor] = None,
        X_observed: Optional[Tensor] = None,
        X_baseline: Optional[Tensor] = None,
        pending_penalty_weight: float = 0.0,
        pending_penalty_beta: float = 10.0,
        observed_penalty_weight: float = 0.0,
        observed_penalty_beta: float = 10.0,
        same_batch_penalty_weight: float = 0.0,
        same_batch_penalty_beta: float = 10.0,
        eps: float = 1e-8,
        **kwargs,
    ) -> None:
        if sampler is None:
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))
        if objective is None:
            raise ValueError("objective must be provided for ordinal utility BO.")
        if q_mode not in {"pointwise", "joint"}:
            raise ValueError("q_mode must be 'pointwise' or 'joint'.")
        if reduction not in {"mean", "sum", "max"}:
            raise ValueError("reduction must be 'mean', 'sum', or 'max'.")

        super().__init__(model=model, sampler=sampler, objective=None, **kwargs)
        self.utility_objective = objective
        self.q_mode = q_mode
        self.reduction = reduction
        self.pending_penalty_weight = float(pending_penalty_weight)
        self.pending_penalty_beta = float(pending_penalty_beta)
        self.observed_penalty_weight = float(observed_penalty_weight)
        self.observed_penalty_beta = float(observed_penalty_beta)
        self.same_batch_penalty_weight = float(same_batch_penalty_weight)
        self.same_batch_penalty_beta = float(same_batch_penalty_beta)
        self.eps = float(eps)

        observed = X_observed if X_observed is not None else X_baseline
        observed = _resolve_observed_X(model, observed)
        self.X_observed = _coerce_pending_to_tensor(observed)
        self.set_X_pending(X_pending)

    def set_X_pending(self, X_pending: Optional[Tensor] = None) -> None:
        self.X_pending = _coerce_pending_to_tensor(X_pending)

    def _posterior_samples_as_utility(self, X: Tensor, *, name: str) -> Tensor:
        Xq = ensure_q_batch(X)
        posterior = self.model.posterior(Xq)
        samples = self.get_posterior_samples(posterior)
        try:
            utility = self.utility_objective(samples, X=Xq)
        except TypeError:
            utility = self.utility_objective(samples)
        if not torch.is_tensor(utility):
            raise TypeError(f"{name}: objective must return Tensor.")
        return _normalize_utility_samples(
            utility,
            Xq,
            sampler=self.sampler,
            name=name,
        )

    def _joint_X(self, X: Tensor) -> Tensor:
        Xq = ensure_q_batch(X)
        if self.X_pending is None:
            return Xq
        pending = self.X_pending.to(device=Xq.device, dtype=Xq.dtype)
        return torch.cat([Xq, match_batch_shape(pending, Xq)], dim=-2)

    def _nominal_transformed_X(self, X: Tensor) -> Tensor:
        Xq = ensure_q_batch(X)
        Xt = _apply_input_transform_for_pending(self.model, Xq)
        q = int(Xq.shape[-2])
        if Xt.shape[-2] == q:
            return Xt
        if q > 0 and Xt.shape[-2] % q == 0:
            n_w = Xt.shape[-2] // q
            return Xt.reshape(*Xt.shape[:-2], q, n_w, Xt.shape[-1]).mean(dim=-2)
        raise RuntimeError(
            "Could not reduce transformed q-like axis to nominal q. "
            f"X.shape={tuple(Xq.shape)}, Xt.shape={tuple(Xt.shape)}."
        )

    def _reference_penalty_per_point(
        self,
        X: Tensor,
        X_ref: Optional[Tensor],
        *,
        weight: float,
        beta: float,
    ) -> Tensor:
        Xn = self._nominal_transformed_X(X)
        if weight <= 0.0 or X_ref is None:
            return Xn.new_zeros(Xn.shape[:-1])

        Xr = self._nominal_transformed_X(
            X_ref.to(device=Xn.device, dtype=Xn.dtype)
        ).reshape(-1, Xn.shape[-1])
        batch_shape = Xn.shape[:-2]
        q = int(Xn.shape[-2])
        Xb = Xn.reshape(-1, q, Xn.shape[-1])
        Xr = Xr.view(1, Xr.shape[0], Xr.shape[1]).expand(Xb.shape[0], -1, -1)
        cat_dims = _cat_dims_from_model(self.model, Xn.shape[-1])
        d2 = _pairwise_distance2(Xb, Xr, cat_dims=cat_dims)
        nearest = d2.min(dim=-1).values
        return (weight * torch.exp(-beta * nearest)).reshape(*batch_shape, q)

    def _same_batch_penalty(self, X: Tensor) -> Tensor:
        Xn = self._nominal_transformed_X(X)
        batch_shape = Xn.shape[:-2]
        q = int(Xn.shape[-2])
        if q <= 1 or self.same_batch_penalty_weight <= 0.0:
            return Xn.new_zeros(batch_shape)

        Xb = Xn.reshape(-1, q, Xn.shape[-1])
        cat_dims = _cat_dims_from_model(self.model, Xn.shape[-1])
        d2 = _pairwise_distance2(Xb, Xb, cat_dims=cat_dims)
        eye = torch.eye(q, device=Xn.device, dtype=torch.bool).unsqueeze(0)
        d2 = d2.masked_fill(eye, float("inf"))
        penalty = (
            0.5
            * self.same_batch_penalty_weight
            * torch.exp(-self.same_batch_penalty_beta * d2).sum(dim=(-1, -2))
        )
        return penalty.reshape(*batch_shape)

    def _reduce_q(self, score: Tensor) -> Tensor:
        if self.reduction == "mean":
            return score.mean(dim=-1)
        if self.reduction == "sum":
            return score.sum(dim=-1)
        if self.reduction == "max":
            return score.max(dim=-1).values
        raise ValueError(f"Unknown reduction={self.reduction!r}.")

    def _pointwise_to_value(self, score: Tensor, X: Tensor) -> Tensor:
        score = score - self._reference_penalty_per_point(
            X,
            self.X_pending,
            weight=self.pending_penalty_weight,
            beta=self.pending_penalty_beta,
        )
        score = score - self._reference_penalty_per_point(
            X,
            self.X_observed,
            weight=self.observed_penalty_weight,
            beta=self.observed_penalty_beta,
        )
        value = self._reduce_q(score) - self._same_batch_penalty(X)
        return _finalize_output(value, X, name=self.__class__.__name__)

    def _joint_penalty(self, X: Tensor) -> Tensor:
        pending = self._reference_penalty_per_point(
            X,
            self.X_pending,
            weight=self.pending_penalty_weight,
            beta=self.pending_penalty_beta,
        ).mean(dim=-1)
        observed = self._reference_penalty_per_point(
            X,
            self.X_observed,
            weight=self.observed_penalty_weight,
            beta=self.observed_penalty_beta,
        ).mean(dim=-1)
        return pending + observed + self._same_batch_penalty(X)

    def _resolve_best_f(
        self,
        best_f: float | Tensor | None,
        *,
        best_f_margin: float,
        best_f_quantile: float | None,
    ) -> Tensor:
        if best_f is not None:
            return torch.as_tensor(best_f)
        if self.X_observed is None:
            raise ValueError(
                "best_f is None and baseline inputs could not be resolved."
            )
        with torch.no_grad():
            utility = self._posterior_samples_as_utility(
                self.X_observed,
                name="ordinal best_f",
            )
            point_values = _mean_over_sample_dims(utility, self.sampler)
            point_values = point_values.reshape(-1)
            if best_f_quantile is None:
                value = point_values.max()
            else:
                q = float(best_f_quantile)
                if not 0.0 <= q <= 1.0:
                    raise ValueError("best_f_quantile must be in [0, 1].")
                value = torch.quantile(point_values, q)
            return (value - float(best_f_margin)).detach()


class qOrdinalExpectedUtility(_OrdinalPointwiseUtilityBOBase):
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        raw_X = ensure_q_batch(X)
        eval_X = self._joint_X(raw_X) if self.q_mode == "joint" else raw_X
        utility = self._posterior_samples_as_utility(
            eval_X,
            name="qOrdinalExpectedUtility",
        )
        if self.q_mode == "joint":
            value = _mean_over_sample_dims(
                utility.max(dim=-1).values,
                self.sampler,
            ) - self._joint_penalty(raw_X)
            return _finalize_output(value, raw_X, name=self.__class__.__name__)
        pointwise = _mean_over_sample_dims(utility, self.sampler)
        return self._pointwise_to_value(pointwise, raw_X)


class qOrdinalExpectedImprovement(_OrdinalPointwiseUtilityBOBase):
    def __init__(
        self,
        model: Model,
        best_f: float | Tensor | None = None,
        *,
        best_f_margin: float = 1e-4,
        best_f_quantile: float | None = None,
        **kwargs,
    ) -> None:
        super().__init__(model=model, **kwargs)
        resolved = self._resolve_best_f(
            best_f,
            best_f_margin=best_f_margin,
            best_f_quantile=best_f_quantile,
        )
        self.register_buffer("best_f", resolved)

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        raw_X = ensure_q_batch(X)
        eval_X = self._joint_X(raw_X) if self.q_mode == "joint" else raw_X
        utility = self._posterior_samples_as_utility(
            eval_X,
            name="qOrdinalExpectedImprovement",
        )
        improvement = (utility - self.best_f.to(utility)).clamp_min(0.0)
        if self.q_mode == "joint":
            value = _mean_over_sample_dims(
                improvement.max(dim=-1).values,
                self.sampler,
            ) - self._joint_penalty(raw_X)
            return _finalize_output(value, raw_X, name=self.__class__.__name__)
        pointwise = _mean_over_sample_dims(improvement, self.sampler)
        return self._pointwise_to_value(pointwise, raw_X)


class qOrdinalProbabilityOfImprovement(_OrdinalPointwiseUtilityBOBase):
    def __init__(
        self,
        model: Model,
        best_f: float | Tensor | None = None,
        *,
        tau: float = 1e-3,
        best_f_margin: float = 1e-4,
        best_f_quantile: float | None = None,
        **kwargs,
    ) -> None:
        super().__init__(model=model, **kwargs)
        resolved = self._resolve_best_f(
            best_f,
            best_f_margin=best_f_margin,
            best_f_quantile=best_f_quantile,
        )
        self.register_buffer("best_f", resolved)
        self.register_buffer("tau", torch.as_tensor(tau))

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        raw_X = ensure_q_batch(X)
        eval_X = self._joint_X(raw_X) if self.q_mode == "joint" else raw_X
        utility = self._posterior_samples_as_utility(
            eval_X,
            name="qOrdinalProbabilityOfImprovement",
        )
        indicator = torch.sigmoid(
            (utility - self.best_f.to(utility))
            / self.tau.to(utility).clamp_min(self.eps)
        )
        if self.q_mode == "joint":
            value = _mean_over_sample_dims(
                indicator.max(dim=-1).values,
                self.sampler,
            ) - self._joint_penalty(raw_X)
            return _finalize_output(value, raw_X, name=self.__class__.__name__)
        pointwise = _mean_over_sample_dims(indicator, self.sampler)
        return self._pointwise_to_value(pointwise, raw_X)


class qOrdinalUpperConfidenceBound(_OrdinalPointwiseUtilityBOBase):
    def __init__(
        self,
        model: Model,
        beta: float | Tensor = 2.0,
        **kwargs,
    ) -> None:
        super().__init__(model=model, **kwargs)
        self.register_buffer("beta", torch.as_tensor(beta))

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        raw_X = ensure_q_batch(X)
        eval_X = self._joint_X(raw_X) if self.q_mode == "joint" else raw_X
        utility = self._posterior_samples_as_utility(
            eval_X,
            name="qOrdinalUpperConfidenceBound",
        )
        sample_ndim = len(getattr(self.sampler, "sample_shape", torch.Size([1])))
        sample_dims = tuple(range(sample_ndim))
        mean = utility.mean(dim=sample_dims, keepdim=True)
        beta_prime = torch.sqrt(
            self.beta.to(utility) * utility.new_tensor(math.pi / 2.0)
        )
        sample_ucb = mean + beta_prime * (utility - mean).abs()
        if self.q_mode == "joint":
            value = _mean_over_sample_dims(
                sample_ucb.max(dim=-1).values,
                self.sampler,
            ) - self._joint_penalty(raw_X)
            return _finalize_output(value, raw_X, name=self.__class__.__name__)
        pointwise = _mean_over_sample_dims(sample_ucb, self.sampler)
        return self._pointwise_to_value(pointwise, raw_X)


__all__ = [
    "OrdinalQBatchMode",
    "OrdinalQReduction",
    "qOrdinalExpectedUtility",
    "qOrdinalExpectedImprovement",
    "qOrdinalProbabilityOfImprovement",
    "qOrdinalUpperConfidenceBound",
]
