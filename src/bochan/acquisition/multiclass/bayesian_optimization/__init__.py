"""Multiclass Bayesian optimization acquisitions.

Importing this module is side-effect free. Input-perturbation, baseline, and
constraint behavior must be expressed through models, objectives, or explicit
acquisition constructor arguments rather than runtime class patching.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch
from botorch.acquisition.multi_objective.objective import MCMultiOutputObjective
from botorch.models.model import Model
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.objective import compute_smoothed_feasibility_indicator
from botorch.utils.safe_math import fatmoid
from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor

from bochan.acquisition.multiclass.base import ClassReductionType
from bochan.acquisition.objective.outcome_constraints import (
    OutcomeConstraint,
    split_outcome_constraints,
    wrap_objective_space_constraints,
)

from . import multi_output as _multi_output
from .hetero_multi_output import (
    qHeteroMultiOutputMulticlassExpectedHypervolumeImprovement,
    qHeteroMultiOutputMulticlassNoisyExpectedHypervolumeImprovement,
    qHeteroMultiOutputMulticlassNParEGO,
)
from .hetero_single_output import NoiseCombineType, NoiseWeightMode
from .multi_output import (
    MulticlassTargetProbabilityObjective,
    OutputReductionType,
    compute_observed_multiclass_target_probability_values,
    compute_observed_multiclass_utility,
)
from .multi_output import (
    qMultiOutputMulticlassExpectedHypervolumeImprovement as _BaseMulticlassEHVI,
)
from .multi_output import (
    qMultiOutputMulticlassNoisyExpectedHypervolumeImprovement as _BaseMulticlassNEHVI,
)
from .multi_output import (
    qMultiOutputMulticlassNParEGO as _BaseMultiOutputMulticlassNParEGO,
)
from .nominal_duplicate_safe import (
    qHeteroMulticlassExpectedImprovement,
    qHeteroMulticlassProbabilityOfFeasibility,
    qHeteroMulticlassProbabilityOfImprovement,
    qHeteroMulticlassUpperConfidenceBound,
    qMulticlassProbabilityOfFeasibility,
    qMultiOutputMulticlassExpectedImprovement,
    qMultiOutputMulticlassProbabilityOfFeasibility,
    qMultiOutputMulticlassProbabilityOfImprovement,
    qMultiOutputMulticlassUpperConfidenceBound,
)
from .single_output import (
    compute_multiclass_target_probability_best_f,
    compute_multiclass_target_probability_values,
)
from .standard import (
    MulticlassProbabilityObjective,
    qMulticlassExpectedImprovement,
    qMulticlassProbabilityOfImprovement,
    qMulticlassUpperConfidenceBound,
)


def _normalize_constraint_eta(
    eta: Tensor | float,
    constraints: Sequence[Callable[[Tensor], Tensor]] | None,
) -> Tensor:
    """Normalize BoTorch feasibility temperature to its registered buffer shape."""

    count = 0 if constraints is None else len(constraints)
    if torch.is_tensor(eta):
        eta_tensor = eta
        if eta_tensor.ndim == 0 and count:
            eta_tensor = eta_tensor.expand(count).clone()
        return eta_tensor
    if count:
        return torch.full((count,), float(eta))
    return torch.as_tensor(float(eta))


def _constraint_parameter_list(
    value: Tensor | float | bool | list[Any],
    *,
    count: int,
) -> list[Any]:
    """Normalize scalar and per-constraint parameters to a Python list."""

    if isinstance(value, list):
        if len(value) != count:
            raise ValueError(
                "Constraint parameter length must match the number of constraints."
            )
        return value
    if torch.is_tensor(value):
        if value.ndim == 0:
            return [value for _ in range(count)]
        if value.numel() != count:
            raise ValueError(
                "Constraint parameter length must match the number of constraints."
            )
        return list(value.unbind())
    return [value for _ in range(count)]


def _multiclass_feasibility_weights(
    *,
    constraints: Sequence[Callable[[Tensor], Tensor]],
    samples: Tensor,
    eta: Tensor | float,
    fat: list[bool | None] | bool,
) -> Tensor:
    """Compute feasibility while keeping the multiclass class axis separate from q."""

    eta_values = _constraint_parameter_list(eta, count=len(constraints))
    fat_values = _constraint_parameter_list(fat, count=len(constraints))
    feasibility: Tensor | None = None

    for constraint, eta_value, fat_value in zip(
        constraints,
        eta_values,
        fat_values,
        strict=True,
    ):
        constraint_value = constraint(samples)
        if fat_value is None:
            weight = constraint_value
        else:
            eta_tensor = torch.as_tensor(
                eta_value,
                dtype=constraint_value.dtype,
                device=constraint_value.device,
            )
            if fat_value:
                weight = fatmoid(-constraint_value, tau=eta_tensor)
            else:
                weight = torch.sigmoid(-constraint_value / eta_tensor)
        feasibility = weight if feasibility is None else feasibility * weight

    if feasibility is None:
        raise RuntimeError("At least one constraint is required.")
    return feasibility


class _MulticlassConstraintEHVI:
    """Shape-safe qEHVI feasibility calculation for multiclass raw samples."""

    def _compute_qehvi(
        self,
        samples: Tensor,
        X: Tensor | None = None,
    ) -> Tensor:
        from .input_perturbation import validate_hypervolume_objective_q

        obj = self.objective(samples, X=X)
        validate_hypervolume_objective_q(obj, X)
        q = obj.shape[-2]
        feasibility = None
        if self.constraints is not None:
            feasibility = _multiclass_feasibility_weights(
                constraints=self.constraints,
                samples=samples,
                eta=self.eta,
                fat=self.fat,
            )

        device = self.ref_point.device
        q_subset_indices = self.compute_q_subset_indices(q_out=q, device=device)
        batch_shape = obj.shape[:-2]
        areas_per_segment = torch.zeros(
            *batch_shape,
            self.cell_lower_bounds.shape[-2],
            dtype=obj.dtype,
            device=device,
        )
        cell_batch_ndim = self.cell_lower_bounds.ndim - 2
        sample_batch_view_shape = torch.Size(
            [
                batch_shape[0] if cell_batch_ndim > 0 else 1,
                *[
                    1
                    for _ in range(
                        len(batch_shape) - max(cell_batch_ndim, 1)
                    )
                ],
                *self.cell_lower_bounds.shape[1:-2],
            ]
        )
        view_shape = (
            *sample_batch_view_shape,
            self.cell_upper_bounds.shape[-2],
            1,
            self.cell_upper_bounds.shape[-1],
        )

        for subset_size in range(1, self.q_out + 1):
            subset_indices = q_subset_indices[f"q_choose_{subset_size}"]
            obj_subsets = obj.index_select(
                dim=-2,
                index=subset_indices.view(-1),
            )
            obj_subsets = obj_subsets.view(
                obj.shape[:-2] + subset_indices.shape + obj.shape[-1:]
            )
            overlap_vertices = obj_subsets.min(dim=-2).values
            overlap_vertices = torch.min(
                overlap_vertices.unsqueeze(-3),
                self.cell_upper_bounds.view(view_shape),
            )
            lengths = (
                overlap_vertices - self.cell_lower_bounds.view(view_shape)
            ).clamp_min(0.0)
            areas = lengths.prod(dim=-1)
            if feasibility is not None:
                feasibility_subsets = feasibility.index_select(
                    dim=-1,
                    index=subset_indices.view(-1),
                ).view(feasibility.shape[:-1] + subset_indices.shape)
                areas = areas * feasibility_subsets.unsqueeze(-3).prod(dim=-1)
            areas = areas.sum(dim=-1)
            areas_per_segment += (-1) ** (subset_size + 1) * areas

        return areas_per_segment.sum(dim=-1).mean(dim=0)


class qMultiOutputMulticlassExpectedHypervolumeImprovement(
    _MulticlassConstraintEHVI,
    _BaseMulticlassEHVI,
):
    """Multiclass qEHVI with explicit objective-space constraint handling."""

    def __init__(
        self,
        model: Model,
        ref_point: Tensor | Sequence[float],
        partitioning,
        *,
        target_class: int | Sequence[int] | None = None,
        output_target_classes: Sequence[int] | None = None,
        class_reduction: ClassReductionType = "mean",
        utility_values: Sequence[Sequence[float]] | Sequence[float] | Tensor | None = None,
        objective_signs: Sequence[float] | Tensor | None = None,
        sampler: SobolQMCNormalSampler | None = None,
        objective: MCMultiOutputObjective | None = None,
        constraints: Sequence[Callable[[Tensor], Tensor]] | None = None,
        X_pending: Tensor | None = None,
        eta: float | Tensor = 1e-3,
        fat: bool = False,
        eps: float = 1e-8,
    ) -> None:
        adapted_constraints = wrap_objective_space_constraints(
            constraints,
            objective_getter=lambda: getattr(self, "objective", None),
        )
        eta_tensor = _normalize_constraint_eta(eta, adapted_constraints)
        super().__init__(
            model=model,
            ref_point=ref_point,
            partitioning=partitioning,
            target_class=target_class,
            output_target_classes=output_target_classes,
            class_reduction=class_reduction,
            utility_values=utility_values,
            objective_signs=objective_signs,
            sampler=sampler,
            objective=objective,
            constraints=adapted_constraints,
            X_pending=X_pending,
            eta=eta_tensor,
            fat=fat,
            eps=eps,
        )


class qMultiOutputMulticlassNoisyExpectedHypervolumeImprovement(
    _MulticlassConstraintEHVI,
    _BaseMulticlassNEHVI,
):
    """Multiclass qNEHVI with explicit objective-space constraint handling."""

    def __init__(
        self,
        model: Model,
        ref_point: Tensor | Sequence[float],
        X_baseline: Tensor,
        *,
        target_class: int | Sequence[int] | None = None,
        output_target_classes: Sequence[int] | None = None,
        class_reduction: ClassReductionType = "mean",
        utility_values: Sequence[Sequence[float]] | Sequence[float] | Tensor | None = None,
        objective_signs: Sequence[float] | Tensor | None = None,
        sampler: SobolQMCNormalSampler | None = None,
        objective: MCMultiOutputObjective | None = None,
        constraints: Sequence[Callable[[Tensor], Tensor]] | None = None,
        X_pending: Tensor | None = None,
        eta: float | Tensor = 1e-3,
        fat: bool = False,
        prune_baseline: bool = False,
        alpha: float = 0.0,
        cache_pending: bool = True,
        max_iep: int = 0,
        incremental_nehvi: bool = True,
        cache_root: bool = False,
        marginalize_dim: int | None = None,
        eps: float = 1e-8,
    ) -> None:
        adapted_constraints = wrap_objective_space_constraints(
            constraints,
            objective_getter=lambda: getattr(self, "objective", None),
        )
        eta_tensor = _normalize_constraint_eta(eta, adapted_constraints)
        super().__init__(
            model=model,
            ref_point=ref_point,
            X_baseline=X_baseline,
            target_class=target_class,
            output_target_classes=output_target_classes,
            class_reduction=class_reduction,
            utility_values=utility_values,
            objective_signs=objective_signs,
            sampler=sampler,
            objective=objective,
            constraints=adapted_constraints,
            X_pending=X_pending,
            eta=eta_tensor,
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


class qMultiOutputMulticlassNParEGO(_BaseMultiOutputMulticlassNParEGO):
    """Multiclass NParEGO with explicit BoTorch-style outcome constraints."""

    def __init__(
        self,
        *args,
        constraints: Sequence[OutcomeConstraint] | None = None,
        eta: float | Tensor = 1e-3,
        fat: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        raw_constraints, objective_constraints = split_outcome_constraints(constraints)
        self.constraints = list(constraints or [])
        self.raw_constraints = raw_constraints
        self.objective_constraints = objective_constraints
        self.eta = eta
        self.fat = bool(fat)

    def _feasibility_factor(
        self,
        *,
        raw_samples: Tensor,
        objective_values: Tensor,
    ) -> Tensor | None:
        factor: Tensor | None = None
        if self.raw_constraints:
            factor = compute_smoothed_feasibility_indicator(
                constraints=self.raw_constraints,
                samples=raw_samples,
                eta=self.eta,
                fat=self.fat,
            )
        if self.objective_constraints:
            objective_factor = compute_smoothed_feasibility_indicator(
                constraints=self.objective_constraints,
                samples=objective_values,
                eta=self.eta,
                fat=self.fat,
            )
            factor = objective_factor if factor is None else factor * objective_factor
        return factor

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        Xq = _multi_output.ensure_q_batch(X)
        posterior = self.model.posterior(Xq)
        samples = self.get_posterior_samples(posterior)
        objective_values = self.base_objective(samples, X=Xq)
        scalarized = self._scalarize(objective_values)
        improvement = (
            scalarized - self.best_value.to(scalarized)
        ).clamp_min(0.0)
        feasibility = self._feasibility_factor(
            raw_samples=samples,
            objective_values=objective_values,
        )
        if feasibility is not None:
            improvement = improvement * feasibility
        return _multi_output._reduce_sample_and_q_to_tbatch(improvement, Xq)


class qMulticlassExpectedHypervolumeImprovement(
    qMultiOutputMulticlassExpectedHypervolumeImprovement
):
    """Multiclass multi-objective qEHVI with domain-first naming."""


class qMulticlassNoisyExpectedHypervolumeImprovement(
    qMultiOutputMulticlassNoisyExpectedHypervolumeImprovement
):
    """Multiclass multi-objective qNEHVI with domain-first naming."""


class qMulticlassNParEGO(qMultiOutputMulticlassNParEGO):
    """Multiclass multi-objective NParEGO with domain-first naming."""


class qHeteroMulticlassExpectedHypervolumeImprovement(
    qHeteroMultiOutputMulticlassExpectedHypervolumeImprovement
):
    """Heteroscedastic multiclass qEHVI with domain-first naming."""


class qHeteroMulticlassNoisyExpectedHypervolumeImprovement(
    qHeteroMultiOutputMulticlassNoisyExpectedHypervolumeImprovement
):
    """Heteroscedastic multiclass qNEHVI with domain-first naming."""


class qHeteroMulticlassNParEGO(qHeteroMultiOutputMulticlassNParEGO):
    """Heteroscedastic multiclass NParEGO with domain-first naming."""


__all__ = [
    "MulticlassProbabilityObjective",
    "MulticlassTargetProbabilityObjective",
    "NoiseCombineType",
    "NoiseWeightMode",
    "OutputReductionType",
    "compute_multiclass_target_probability_best_f",
    "compute_multiclass_target_probability_values",
    "compute_observed_multiclass_target_probability_values",
    "compute_observed_multiclass_utility",
    "qHeteroMulticlassExpectedHypervolumeImprovement",
    "qHeteroMulticlassExpectedImprovement",
    "qHeteroMulticlassNParEGO",
    "qHeteroMulticlassNoisyExpectedHypervolumeImprovement",
    "qHeteroMulticlassProbabilityOfFeasibility",
    "qHeteroMulticlassProbabilityOfImprovement",
    "qHeteroMulticlassUpperConfidenceBound",
    "qMulticlassExpectedHypervolumeImprovement",
    "qMulticlassExpectedImprovement",
    "qMulticlassNParEGO",
    "qMulticlassNoisyExpectedHypervolumeImprovement",
    "qMulticlassProbabilityOfFeasibility",
    "qMulticlassProbabilityOfImprovement",
    "qMulticlassUpperConfidenceBound",
    "qMultiOutputMulticlassExpectedHypervolumeImprovement",
    "qMultiOutputMulticlassExpectedImprovement",
    "qMultiOutputMulticlassNParEGO",
    "qMultiOutputMulticlassNoisyExpectedHypervolumeImprovement",
    "qMultiOutputMulticlassProbabilityOfFeasibility",
    "qMultiOutputMulticlassProbabilityOfImprovement",
    "qMultiOutputMulticlassUpperConfidenceBound",
]
