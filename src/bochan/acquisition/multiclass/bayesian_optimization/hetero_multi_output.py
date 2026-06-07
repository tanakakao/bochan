from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import torch
from botorch.acquisition.multi_objective.objective import MCMultiOutputObjective
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning
from torch import Tensor

from bochan.acquisition.multiclass.active_learning.hetero_multi_output import _HeteroMultiOutputMulticlassMixin
from bochan.acquisition.multiclass.base import ClassReductionType

from .multi_output import (
    MulticlassTargetProbabilityObjective,
    qMultiOutputMulticlassExpectedHypervolumeImprovement,
    qMultiOutputMulticlassExpectedImprovement,
    qMultiOutputMulticlassNParEGO,
    qMultiOutputMulticlassNoisyExpectedHypervolumeImprovement,
    qMultiOutputMulticlassProbabilityOfFeasibility,
    qMultiOutputMulticlassProbabilityOfImprovement,
    qMultiOutputMulticlassUpperConfidenceBound,
)


class _HeteroMulticlassTargetProbabilityObjective(MCMultiOutputObjective):
    """Target-class objective with heteroscedastic noise weighting.

    This mirrors the binary hetero multi-output design: BoTorch EHVI / NEHVI
    still perform the acquisition computation, while the objective maps raw
    posterior samples to target-class probabilities and applies a noise-aware
    weighting in objective space.
    """

    def __init__(
        self,
        *,
        base_objective: MCMultiOutputObjective,
        model,
        noise_mode: str = "inverse_linear",
        noise_combine: str = "multiply",
        noise_penalty_lambda: float = 1.0,
        noise_min_weight: float = 0.0,
        noise_weight_scale: float = 1.0,
        noise_model_outputs_log_var: bool = True,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.base_objective = base_objective
        self.model = model
        self.noise_mode = noise_mode
        self.noise_combine = noise_combine
        self.noise_penalty_lambda = float(noise_penalty_lambda)
        self.noise_min_weight = float(noise_min_weight)
        self.noise_weight_scale = float(noise_weight_scale)
        self.noise_model_outputs_log_var = bool(noise_model_outputs_log_var)
        self.eps = float(eps)

    @staticmethod
    def _ensure_q_batch(X: Tensor) -> Tensor:
        if X.ndim == 1:
            return X.view(1, 1, -1)
        if X.ndim == 2:
            return X.unsqueeze(0)
        return X

    def _submodels(self) -> list:
        submodels = getattr(self.model, "models", None)
        if submodels is not None:
            return list(submodels)
        submodels = getattr(self.model, "submodels", None)
        if submodels is not None:
            return list(submodels)
        return []

    def _single_noise(self, model, X: Tensor) -> Tensor:
        fn = getattr(model, "predict_noise_var", None)
        if callable(fn):
            return torch.as_tensor(fn(X), device=X.device, dtype=X.dtype)
        for name in ("posterior_noise", "noise_posterior"):
            fn = getattr(model, name, None)
            if callable(fn):
                return torch.as_tensor(fn(X).mean, device=X.device, dtype=X.dtype)
        noise_model = getattr(model, "noise_model", None)
        if noise_model is None:
            inner = getattr(model, "model", None)
            noise_model = getattr(inner, "noise_model", None) if inner is not None else None
        if noise_model is not None:
            return torch.as_tensor(noise_model.posterior(X).mean, device=X.device, dtype=X.dtype)
        return torch.zeros(X.shape[:-1], device=X.device, dtype=X.dtype)

    def _to_point_noise(self, noise: Tensor, X: Tensor) -> Tensor:
        point_shape = X.shape[:-1]
        while noise.ndim > len(point_shape):
            if noise.shape[-1] == 1:
                noise = noise.squeeze(-1)
            else:
                noise = noise.mean(dim=-1)
        if noise.shape == point_shape:
            return noise
        if noise.numel() == int(torch.tensor(point_shape).prod().item()):
            return noise.reshape(point_shape)
        return noise.mean().expand(point_shape)

    def _noise_values(self, X: Tensor, n_outputs: int) -> Tensor:
        Xq = self._ensure_q_batch(X)
        if self.noise_mode == "none":
            return torch.zeros(*Xq.shape[:-1], n_outputs, device=Xq.device, dtype=Xq.dtype)

        fn = getattr(self.model, "predict_noise_var", None)
        if callable(fn):
            noise = torch.as_tensor(fn(Xq), device=Xq.device, dtype=Xq.dtype)
            if noise.shape == (*Xq.shape[:-1], n_outputs):
                out = noise
            else:
                out = noise.reshape(*Xq.shape[:-1], n_outputs) if noise.numel() == int(torch.tensor((*Xq.shape[:-1], n_outputs)).prod().item()) else noise.mean().expand(*Xq.shape[:-1], n_outputs)
        else:
            pieces = []
            submodels = self._submodels()
            if len(submodels) == 0:
                submodels = [self.model]
            for sm in submodels:
                pieces.append(self._to_point_noise(self._single_noise(sm, Xq), Xq).unsqueeze(-1))
            out = torch.cat(pieces, dim=-1)
            if out.shape[-1] != n_outputs:
                out = out.mean(dim=-1, keepdim=True).expand(*Xq.shape[:-1], n_outputs)

        if self.noise_model_outputs_log_var:
            return torch.exp(out.clamp(min=-30.0, max=30.0)).clamp_min(self.eps)
        return out.clamp_min(self.eps)

    def _noise_weight(self, noise: Tensor) -> Tensor:
        if self.noise_mode == "none":
            weight = torch.ones_like(noise)
        elif self.noise_mode == "inverse_linear":
            weight = 1.0 / (1.0 + self.noise_penalty_lambda * noise.clamp_min(0.0))
        elif self.noise_mode == "inverse_sqrt":
            weight = 1.0 / torch.sqrt(1.0 + self.noise_penalty_lambda * noise.clamp_min(0.0))
        elif self.noise_mode == "exp":
            weight = torch.exp(-self.noise_penalty_lambda * noise.clamp_min(0.0))
        else:
            raise ValueError(f"Unknown noise_mode: {self.noise_mode!r}.")
        return (self.noise_weight_scale * weight).clamp_min(self.noise_min_weight)

    def _combine(self, values: Tensor, weight: Tensor) -> Tensor:
        # values: sample_shape x batch_shape x q x m
        while weight.ndim < values.ndim:
            weight = weight.unsqueeze(0)
        weight = weight.to(device=values.device, dtype=values.dtype)
        if self.noise_combine == "multiply":
            return values * weight
        if self.noise_combine in {"subtract", "add"}:
            return values - (1.0 - weight)
        raise ValueError(f"Unknown noise_combine: {self.noise_combine!r}.")

    def forward(self, samples: Tensor, X: Optional[Tensor] = None) -> Tensor:
        values = self.base_objective(samples, X=X)
        if X is None:
            return values
        if values.ndim < 1:
            return values
        n_outputs = int(values.shape[-1])
        noise = self._noise_values(X, n_outputs=n_outputs)
        weight = self._noise_weight(noise)
        return self._combine(values, weight)


class qHeteroMultiOutputMulticlassProbabilityOfFeasibility(
    _HeteroMultiOutputMulticlassMixin,
    qMultiOutputMulticlassProbabilityOfFeasibility,
):
    pass


class qHeteroMultiOutputMulticlassExpectedHypervolumeImprovement(qMultiOutputMulticlassExpectedHypervolumeImprovement):
    """BoTorch qEHVI with hetero-adjusted multiclass target objective."""

    def __init__(
        self,
        model,
        ref_point: Tensor | Sequence[float],
        partitioning: FastNondominatedPartitioning,
        *,
        target_class: int | Sequence[int] | None = None,
        output_target_classes: Sequence[int] | None = None,
        class_reduction: ClassReductionType = "mean",
        sampler: Optional[SobolQMCNormalSampler] = None,
        objective: Optional[MCMultiOutputObjective] = None,
        constraints: Optional[list] = None,
        X_pending: Optional[Tensor] = None,
        eta: float | Tensor = 1e-3,
        fat: bool = False,
        noise_mode: str = "inverse_linear",
        noise_combine: str = "multiply",
        noise_penalty_lambda: float = 1.0,
        noise_min_weight: float = 0.0,
        noise_weight_scale: float = 1.0,
        noise_model_outputs_log_var: bool = True,
        eps: float = 1e-8,
    ) -> None:
        base_objective = objective or MulticlassTargetProbabilityObjective(
            target_class=target_class,
            output_target_classes=output_target_classes,
            num_outputs=int(torch.as_tensor(ref_point).numel()),
            class_reduction=class_reduction,
            eps=eps,
        )
        hetero_objective = _HeteroMulticlassTargetProbabilityObjective(
            base_objective=base_objective,
            model=model,
            noise_mode=noise_mode,
            noise_combine=noise_combine,
            noise_penalty_lambda=noise_penalty_lambda,
            noise_min_weight=noise_min_weight,
            noise_weight_scale=noise_weight_scale,
            noise_model_outputs_log_var=noise_model_outputs_log_var,
            eps=eps,
        )
        super().__init__(
            model=model,
            ref_point=ref_point,
            partitioning=partitioning,
            sampler=sampler,
            objective=hetero_objective,
            constraints=constraints,
            X_pending=X_pending,
            eta=eta,
            fat=fat,
            eps=eps,
        )
        self.base_objective = base_objective
        self.hetero_objective = hetero_objective


class qHeteroMultiOutputMulticlassNoisyExpectedHypervolumeImprovement(qMultiOutputMulticlassNoisyExpectedHypervolumeImprovement):
    """BoTorch qNEHVI with hetero-adjusted multiclass target objective."""

    def __init__(
        self,
        model,
        ref_point: Tensor | Sequence[float],
        X_baseline: Tensor,
        *,
        target_class: int | Sequence[int] | None = None,
        output_target_classes: Sequence[int] | None = None,
        class_reduction: ClassReductionType = "mean",
        sampler: Optional[SobolQMCNormalSampler] = None,
        objective: Optional[MCMultiOutputObjective] = None,
        constraints: Optional[list] = None,
        X_pending: Optional[Tensor] = None,
        eta: float | Tensor = 1e-3,
        fat: bool = False,
        prune_baseline: bool = False,
        alpha: float = 0.0,
        cache_pending: bool = True,
        max_iep: int = 0,
        incremental_nehvi: bool = True,
        cache_root: bool = True,
        marginalize_dim: Optional[int] = None,
        noise_mode: str = "inverse_linear",
        noise_combine: str = "multiply",
        noise_penalty_lambda: float = 1.0,
        noise_min_weight: float = 0.0,
        noise_weight_scale: float = 1.0,
        noise_model_outputs_log_var: bool = True,
        eps: float = 1e-8,
    ) -> None:
        base_objective = objective or MulticlassTargetProbabilityObjective(
            target_class=target_class,
            output_target_classes=output_target_classes,
            num_outputs=int(torch.as_tensor(ref_point).numel()),
            class_reduction=class_reduction,
            eps=eps,
        )
        hetero_objective = _HeteroMulticlassTargetProbabilityObjective(
            base_objective=base_objective,
            model=model,
            noise_mode=noise_mode,
            noise_combine=noise_combine,
            noise_penalty_lambda=noise_penalty_lambda,
            noise_min_weight=noise_min_weight,
            noise_weight_scale=noise_weight_scale,
            noise_model_outputs_log_var=noise_model_outputs_log_var,
            eps=eps,
        )
        super().__init__(
            model=model,
            ref_point=ref_point,
            X_baseline=X_baseline,
            sampler=sampler,
            objective=hetero_objective,
            constraints=constraints,
            X_pending=X_pending,
            eta=eta,
            fat=fat,
            prune_baseline=prune_baseline,
            alpha=alpha,
            cache_pending=cache_pending,
            max_iep=max_iep,
            incremental_nehvi=incremental_nehvi,
            cache_root=cache_root,
            marginalize_dim=marginalize_dim,
            eps=eps,
        )
        self.base_objective = base_objective
        self.hetero_objective = hetero_objective


class qHeteroMultiOutputMulticlassNParEGO(qMultiOutputMulticlassNParEGO):
    """qNParEGO using a hetero-adjusted multiclass target objective."""

    def __init__(
        self,
        model,
        X_baseline: Tensor,
        ref_point: Tensor | Sequence[float],
        *,
        target_class: int | Sequence[int] | None = None,
        output_target_classes: Sequence[int] | None = None,
        class_reduction: ClassReductionType = "mean",
        weights: Optional[Tensor] = None,
        sampler: Optional[SobolQMCNormalSampler] = None,
        objective: Optional[MCMultiOutputObjective] = None,
        rho: float = 0.05,
        noise_mode: str = "inverse_linear",
        noise_combine: str = "multiply",
        noise_penalty_lambda: float = 1.0,
        noise_min_weight: float = 0.0,
        noise_weight_scale: float = 1.0,
        noise_model_outputs_log_var: bool = True,
        eps: float = 1e-8,
    ) -> None:
        ref_tensor = torch.as_tensor(ref_point, device=X_baseline.device, dtype=X_baseline.dtype).reshape(-1)
        base_objective = objective or MulticlassTargetProbabilityObjective(
            target_class=target_class,
            output_target_classes=output_target_classes,
            num_outputs=int(ref_tensor.numel()),
            class_reduction=class_reduction,
            eps=eps,
        )
        hetero_objective = _HeteroMulticlassTargetProbabilityObjective(
            base_objective=base_objective,
            model=model,
            noise_mode=noise_mode,
            noise_combine=noise_combine,
            noise_penalty_lambda=noise_penalty_lambda,
            noise_min_weight=noise_min_weight,
            noise_weight_scale=noise_weight_scale,
            noise_model_outputs_log_var=noise_model_outputs_log_var,
            eps=eps,
        )
        super().__init__(
            model=model,
            X_baseline=X_baseline,
            ref_point=ref_tensor,
            weights=weights,
            sampler=sampler,
            objective=hetero_objective,
            rho=rho,
            eps=eps,
        )
        self.base_objective = base_objective
        self.hetero_objective = hetero_objective


class qHeteroMultiOutputMulticlassExpectedImprovement(
    _HeteroMultiOutputMulticlassMixin,
    qMultiOutputMulticlassExpectedImprovement,
):
    pass


class qHeteroMultiOutputMulticlassProbabilityOfImprovement(
    _HeteroMultiOutputMulticlassMixin,
    qMultiOutputMulticlassProbabilityOfImprovement,
):
    pass


class qHeteroMultiOutputMulticlassUpperConfidenceBound(
    _HeteroMultiOutputMulticlassMixin,
    qMultiOutputMulticlassUpperConfidenceBound,
):
    pass


__all__ = [
    "qHeteroMultiOutputMulticlassProbabilityOfFeasibility",
    "qHeteroMultiOutputMulticlassExpectedHypervolumeImprovement",
    "qHeteroMultiOutputMulticlassNoisyExpectedHypervolumeImprovement",
    "qHeteroMultiOutputMulticlassNParEGO",
    "qHeteroMultiOutputMulticlassExpectedImprovement",
    "qHeteroMultiOutputMulticlassProbabilityOfImprovement",
    "qHeteroMultiOutputMulticlassUpperConfidenceBound",
]
