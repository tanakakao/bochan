from __future__ import annotations

from typing import Literal, Optional

import torch
from torch import Tensor

from botorch.utils.transforms import t_batch_mode_transform

from .deepgp_single_output import QReduceType, _DeepPosteriorAcquisitionBase


MultiOutputAggregation = Literal[
    "weighted_sum",
    "weighted_mean",
    "mean",
    "sum",
    "max",
    "min",
]


class _DeepMultiOutputAcquisitionBase(_DeepPosteriorAcquisitionBase):
    """
    多出力 DeepGP / DeepMixedGP 向けの共通基底。

    前提
    ----
    ``model.posterior(X).mean`` / ``variance`` が ``[..., q, m]`` を返すこと。
    """

    def __init__(
        self,
        model,
        aggregation: MultiOutputAggregation = "weighted_mean",
        output_weights: Optional[Tensor] = None,
        normalize_output_weights: bool = True,
        q_reduction: QReduceType = "mean",
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
        eps: float = 1e-12,
    ) -> None:
        super().__init__(
            model=model,
            q_reduction=q_reduction,
            output_reduction="mean",
            X_pending=X_pending,
            X_observed=X_observed,
            same_batch_penalty_weight=same_batch_penalty_weight,
            same_batch_penalty_beta=same_batch_penalty_beta,
            pending_penalty_weight=pending_penalty_weight,
            pending_penalty_beta=pending_penalty_beta,
            observed_penalty_weight=observed_penalty_weight,
            observed_penalty_beta=observed_penalty_beta,
            hard_duplicate_penalty=hard_duplicate_penalty,
            hard_duplicate_tol=hard_duplicate_tol,
            eps=eps,
        )
        self.aggregation = aggregation
        self.normalize_output_weights = bool(normalize_output_weights)
        if output_weights is not None:
            w = output_weights.detach().clone()
            if w.ndim != 1:
                raise ValueError("output_weights must have shape [m].")
            self.register_buffer("output_weights", w)
        else:
            self.output_weights = None

    def _aggregate_outputs(self, t: Tensor) -> Tensor:
        if t.ndim < 2:
            return t

        weights = self.output_weights
        if weights is not None:
            if t.shape[-1] != weights.shape[0]:
                raise ValueError(
                    f"Mismatch between last dim ({t.shape[-1]}) and output_weights ({weights.shape[0]})."
                )
            w = weights.to(device=t.device, dtype=t.dtype)
            if self.normalize_output_weights:
                w = w / w.sum().clamp_min(self.eps)
        else:
            w = None

        if self.aggregation == "weighted_sum":
            if w is None:
                raise ValueError("aggregation='weighted_sum' requires output_weights.")
            return (t * w).sum(dim=-1)
        if self.aggregation == "weighted_mean":
            if w is None:
                return t.mean(dim=-1)
            return (t * w).sum(dim=-1)
        if self.aggregation == "mean":
            return t.mean(dim=-1)
        if self.aggregation == "sum":
            return t.sum(dim=-1)
        if self.aggregation == "max":
            return t.max(dim=-1).values
        if self.aggregation == "min":
            return t.min(dim=-1).values
        raise ValueError(f"Unknown aggregation mode: {self.aggregation}")

    def _posterior_mean_std_multi(self, X: Tensor) -> tuple[Tensor, Tensor]:
        post = self.model.posterior(X, observation_noise=False)
        mean = post.mean
        var = post.variance.clamp_min(self.eps)
        std = var.sqrt()

        if mean.ndim == X.ndim:
            mean = mean.squeeze(-1)
        elif mean.ndim >= 3:
            mean = self._aggregate_outputs(mean)

        if std.ndim == X.ndim:
            std = std.squeeze(-1)
        elif std.ndim >= 3:
            std = self._aggregate_outputs(std)

        return mean, std

    def _reference_variance_multi(self, X_ref: Tensor) -> Tensor:
        post = self.model.posterior(X_ref, observation_noise=False)
        ref_var = post.variance.clamp_min(self.eps)
        if ref_var.ndim == 2:
            ref_var = ref_var.squeeze(-1)
        elif ref_var.ndim >= 3:
            ref_var = self._aggregate_outputs(ref_var)
        return ref_var


class qDeepPosteriorVarianceMulti(_DeepMultiOutputAcquisitionBase):
    """
    多出力 DeepGP 用 posterior variance 探索。

    ``score(X) = reduce_q(aggregate_output(var(X))) - penalty(X)``
    """

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        ref_var = self._reference_variance_multi(X)
        score = self._reduce_q(ref_var)
        return score - self._total_penalty(X)


class qDeepStraddleMulti(_DeepMultiOutputAcquisitionBase):
    """
    多出力 DeepGP 用 straddle。

    ``score(X) = reduce_q(aggregate_output(beta * std - |mean - target|)) - penalty(X)``

    target
    ------
    - float: 全出力共通
    - Tensor[m]: 出力ごとに異なる目標値
    """

    def __init__(
        self,
        model,
        target: float | Tensor = 0.0,
        beta: float = 1.96,
        aggregation: MultiOutputAggregation = "weighted_mean",
        output_weights: Optional[Tensor] = None,
        normalize_output_weights: bool = True,
        q_reduction: QReduceType = "mean",
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
        eps: float = 1e-12,
    ) -> None:
        super().__init__(
            model=model,
            aggregation=aggregation,
            output_weights=output_weights,
            normalize_output_weights=normalize_output_weights,
            q_reduction=q_reduction,
            X_pending=X_pending,
            X_observed=X_observed,
            same_batch_penalty_weight=same_batch_penalty_weight,
            same_batch_penalty_beta=same_batch_penalty_beta,
            pending_penalty_weight=pending_penalty_weight,
            pending_penalty_beta=pending_penalty_beta,
            observed_penalty_weight=observed_penalty_weight,
            observed_penalty_beta=observed_penalty_beta,
            hard_duplicate_penalty=hard_duplicate_penalty,
            hard_duplicate_tol=hard_duplicate_tol,
            eps=eps,
        )
        if torch.is_tensor(target):
            self.register_buffer("target", target.detach().clone())
        else:
            self.target = torch.tensor(float(target))
        self.beta = float(beta)

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        post = self.model.posterior(X, observation_noise=False)
        mean = post.mean
        var = post.variance.clamp_min(self.eps)
        std = var.sqrt()

        if mean.ndim == X.ndim:
            mean = mean.squeeze(-1)
            std = std.squeeze(-1)
            score_per_point = self.beta * std - (mean - self.target.to(mean)).abs()
        else:
            target = self.target.to(mean)
            if target.ndim == 0:
                score_raw = self.beta * std - (mean - target).abs()
            else:
                if target.ndim != 1 or target.shape[0] != mean.shape[-1]:
                    raise ValueError(
                        f"target must be scalar or shape [m], but got {tuple(target.shape)}"
                    )
                view_shape = [1] * (mean.ndim - 1) + [-1]
                score_raw = self.beta * std - (mean - target.view(*view_shape)).abs()
            score_per_point = self._aggregate_outputs(score_raw)

        score = self._reduce_q(score_per_point)
        return score - self._total_penalty(X)


class qDeepIntegratedPosteriorVarianceProxyMulti(_DeepMultiOutputAcquisitionBase):
    """
    多出力 DeepGP 向け qNIPV proxy。

    参照点 ``X_ref`` 上の多出力 posterior variance を集約し、
    候補 ``X`` がその高不確実領域をどれだけカバーしているかを評価する。
    """

    def __init__(
        self,
        model,
        X_ref: Tensor,
        kernel_lengthscale: float = 0.2,
        normalize_weights: bool = True,
        aggregation: MultiOutputAggregation = "weighted_mean",
        output_weights: Optional[Tensor] = None,
        normalize_output_weights: bool = True,
        q_reduction: QReduceType = "mean",
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
        eps: float = 1e-12,
    ) -> None:
        super().__init__(
            model=model,
            aggregation=aggregation,
            output_weights=output_weights,
            normalize_output_weights=normalize_output_weights,
            q_reduction=q_reduction,
            X_pending=X_pending,
            X_observed=X_observed,
            same_batch_penalty_weight=same_batch_penalty_weight,
            same_batch_penalty_beta=same_batch_penalty_beta,
            pending_penalty_weight=pending_penalty_weight,
            pending_penalty_beta=pending_penalty_beta,
            observed_penalty_weight=observed_penalty_weight,
            observed_penalty_beta=observed_penalty_beta,
            hard_duplicate_penalty=hard_duplicate_penalty,
            hard_duplicate_tol=hard_duplicate_tol,
            eps=eps,
        )
        if X_ref.ndim != 2:
            raise ValueError("X_ref must have shape [n_ref, d].")
        self.register_buffer("X_ref", X_ref.detach().clone())
        self.kernel_lengthscale = float(kernel_lengthscale)
        self.normalize_weights = bool(normalize_weights)

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        ref_var = self._reference_variance_multi(self.X_ref)
        d2 = self._pairwise_sq_dists(X, self.X_ref)
        ls2 = max(self.kernel_lengthscale ** 2, self.eps)
        weights = torch.exp(-0.5 * d2 / ls2)

        if self.normalize_weights:
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(self.eps)

        local_scores = (weights * ref_var.view(*([1] * (weights.ndim - 1)), -1)).sum(dim=-1)
        score = self._reduce_q(local_scores)
        return score - self._total_penalty(X)


__all__ = [
    "QReduceType",
    "MultiOutputAggregation",
    "qDeepPosteriorVarianceMulti",
    "qDeepStraddleMulti",
    "qDeepIntegratedPosteriorVarianceProxyMulti",
]
