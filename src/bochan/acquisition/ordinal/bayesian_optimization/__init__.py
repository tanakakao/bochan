"""Ordinal Bayesian optimization acquisitions.

Importing this module is side-effect free. Ordinal utility conversion is an
explicit BoTorch objective; acquisition classes are not patched at runtime.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from botorch.acquisition.multi_objective.objective import MCMultiOutputObjective
from botorch.models.model import Model
from botorch.sampling.base import MCSampler
from botorch.utils.objective import compute_smoothed_feasibility_indicator
from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor, nn

from bochan.acquisition._nehvi_cache_root import resolve_nehvi_cache_root
from bochan.acquisition.objective.outcome_constraints import (
    OutcomeConstraint,
    split_outcome_constraints,
    wrap_objective_space_constraints,
)

from . import multi_output as _multi_output
from ._baseline import (
    complete_ordinal_baseline_rows,
    infer_multioutput_ordinal_train_y,
)
from ._utility_defaults import infer_multioutput_ordinal_utility_values
from .hetero_multi_output import (
    qHeteroMultiOutputOrdinalExpectedHypervolumeImprovement,
    qHeteroMultiOutputOrdinalExpectedImprovement,
    qHeteroMultiOutputOrdinalExpectedUtility,
    qHeteroMultiOutputOrdinalNoisyExpectedHypervolumeImprovement,
    qHeteroMultiOutputOrdinalNormalScoreObjective,
    qHeteroMultiOutputOrdinalNParEGO,
    qHeteroMultiOutputOrdinalProbabilityOfImprovement,
)
from .hetero_single_output import (
    qHeteroOrdinalExpectedImprovement,
    qHeteroOrdinalExpectedUtility,
    qHeteroOrdinalExpectedUtilityUpperConfidenceBound,
    qHeteroOrdinalProbabilityOfImprovement,
)
from .knowledge_gradient import qOrdinalKnowledgeGradient
from .multi_output import compute_observed_ordinal_utility
from .multi_output import (
    qMultiOutputOrdinalExpectedHypervolumeImprovement as _BaseOrdinalEHVI,
)
from .multi_output import (
    qMultiOutputOrdinalNoisyExpectedHypervolumeImprovement as _BaseOrdinalNEHVI,
)
from .multi_output import qMultiOutputOrdinalNParEGO as _BaseMultiOutputOrdinalNParEGO
from .multi_output import (
    qMultiOutputOrdinalUtilityObjective as _BaseMultiOutputOrdinalUtilityObjective,
)
from .single_output import (
    compute_ordinal_expected_utility_best_f,
    qOrdinalProbabilityOfFeasibility,
)
from .standard import (
    qOrdinalExpectedImprovement,
    qOrdinalExpectedUtility,
    qOrdinalProbabilityOfImprovement,
    qOrdinalUpperConfidenceBound,
)

_NPAREGO_TBATCH_ERRORS = (
    "Expected scalarized value to have q dimension as the last dimension.",
    "qMultiOutputOrdinalNParEGO produced invalid output shape after scalarization.",
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


def _with_default_utility_values(model, utility_values):
    if utility_values is not None:
        return utility_values
    return infer_multioutput_ordinal_utility_values(model)


class qMultiOutputOrdinalUtilityObjective(_BaseMultiOutputOrdinalUtilityObjective):
    """Ordinal utility objective with explicit wide-multitask likelihood mapping."""

    def __init__(
        self,
        model,
        utility_values=None,
        *,
        ordinal_likelihoods=None,
        **kwargs,
    ) -> None:
        utility_values = _with_default_utility_values(model, utility_values)
        if ordinal_likelihoods is None:
            num_outputs = int(getattr(model, "num_outputs", 1))
            likelihood = getattr(model, "ordinal_likelihood", None)
            if likelihood is None:
                likelihood = getattr(model, "likelihood", None)
            if likelihood is not None and num_outputs > 1:
                ordinal_likelihoods = [likelihood] * num_outputs
        super().__init__(
            model=model,
            utility_values=utility_values,
            ordinal_likelihoods=ordinal_likelihoods,
            **kwargs,
        )


def _resolve_ordinal_objective(
    *,
    model,
    utility_values,
    objective,
    ordinal_likelihoods,
    objective_signs,
    link,
    input_perturbation_n_w,
    risk_type,
    risk_alpha,
):
    if objective is not None:
        return objective
    return qMultiOutputOrdinalUtilityObjective(
        model=model,
        utility_values=utility_values,
        ordinal_likelihoods=ordinal_likelihoods,
        objective_signs=objective_signs,
        link=link,
        input_perturbation_n_w=input_perturbation_n_w,
        risk_type=risk_type,
        risk_alpha=risk_alpha,
    )


class qMultiOutputOrdinalExpectedHypervolumeImprovement(_BaseOrdinalEHVI):
    """Ordinal qEHVI with explicit objective-space constraint handling."""

    def __init__(
        self,
        model: Model,
        ref_point: Sequence[float] | Tensor,
        *,
        partitioning=None,
        utility_values: Sequence[Sequence[float]] | Sequence[float] | Tensor | None = None,
        objective: MCMultiOutputObjective | None = None,
        train_Y: Tensor | None = None,
        Y_baseline: Tensor | None = None,
        ordinal_likelihoods: Sequence[nn.Module] | nn.Module | None = None,
        objective_signs: Sequence[float] | Tensor | None = None,
        class_offset: int = 0,
        sampler: MCSampler | None = None,
        constraints: Sequence[Callable[[Tensor], Tensor]] | None = None,
        X_pending: Tensor | None = None,
        eta: Tensor | float = 1e-3,
        fat: bool = False,
        link: str = "auto",
        input_perturbation_n_w: int | None = None,
        risk_type=None,
        risk_alpha: float = 0.5,
    ) -> None:
        utility_values = _with_default_utility_values(model, utility_values)
        if train_Y is not None:
            train_Y = complete_ordinal_baseline_rows(train_Y)
        objective = _resolve_ordinal_objective(
            model=model,
            utility_values=utility_values,
            objective=objective,
            ordinal_likelihoods=ordinal_likelihoods,
            objective_signs=objective_signs,
            link=link,
            input_perturbation_n_w=input_perturbation_n_w,
            risk_type=risk_type,
            risk_alpha=risk_alpha,
        )
        adapted_constraints = wrap_objective_space_constraints(
            constraints,
            objective_getter=lambda: getattr(self, "objective", None),
        )
        eta_tensor = _normalize_constraint_eta(eta, adapted_constraints)
        super().__init__(
            model=model,
            ref_point=ref_point,
            partitioning=partitioning,
            utility_values=utility_values,
            objective=objective,
            train_Y=train_Y,
            Y_baseline=Y_baseline,
            ordinal_likelihoods=ordinal_likelihoods,
            objective_signs=objective_signs,
            class_offset=class_offset,
            sampler=sampler,
            constraints=adapted_constraints,
            X_pending=X_pending,
            eta=eta_tensor,
            fat=fat,
            link=link,
            input_perturbation_n_w=input_perturbation_n_w,
            risk_type=risk_type,
            risk_alpha=risk_alpha,
        )


class qMultiOutputOrdinalNoisyExpectedHypervolumeImprovement(_BaseOrdinalNEHVI):
    """Ordinal qNEHVI with explicit objective-space constraint handling."""

    def __init__(
        self,
        model: Model,
        ref_point: Sequence[float] | Tensor,
        X_baseline: Tensor,
        *,
        utility_values: Sequence[Sequence[float]] | Sequence[float] | Tensor | None = None,
        objective: MCMultiOutputObjective | None = None,
        ordinal_likelihoods: Sequence[nn.Module] | nn.Module | None = None,
        objective_signs: Sequence[float] | Tensor | None = None,
        sampler: MCSampler | None = None,
        constraints: Sequence[Callable[[Tensor], Tensor]] | None = None,
        X_pending: Tensor | None = None,
        eta: Tensor | float = 1e-3,
        fat: bool = False,
        link: str = "auto",
        input_perturbation_n_w: int | None = None,
        risk_type=None,
        risk_alpha: float = 0.5,
        cache_root: bool | None = None,
        **kwargs,
    ) -> None:
        utility_values = _with_default_utility_values(model, utility_values)
        objective = _resolve_ordinal_objective(
            model=model,
            utility_values=utility_values,
            objective=objective,
            ordinal_likelihoods=ordinal_likelihoods,
            objective_signs=objective_signs,
            link=link,
            input_perturbation_n_w=input_perturbation_n_w,
            risk_type=risk_type,
            risk_alpha=risk_alpha,
        )
        adapted_constraints = wrap_objective_space_constraints(
            constraints,
            objective_getter=lambda: getattr(self, "objective", None),
        )
        eta_tensor = _normalize_constraint_eta(eta, adapted_constraints)
        super().__init__(
            model=model,
            ref_point=ref_point,
            X_baseline=X_baseline,
            utility_values=utility_values,
            objective=objective,
            ordinal_likelihoods=ordinal_likelihoods,
            objective_signs=objective_signs,
            sampler=sampler,
            constraints=adapted_constraints,
            X_pending=X_pending,
            eta=eta_tensor,
            fat=fat,
            link=link,
            input_perturbation_n_w=input_perturbation_n_w,
            risk_type=risk_type,
            risk_alpha=risk_alpha,
            cache_root=resolve_nehvi_cache_root(model, cache_root),
            **kwargs,
        )


class qMultiOutputOrdinalNParEGO(_BaseMultiOutputOrdinalNParEGO):
    """Ordinal NParEGO with explicit baseline, constraints, and t-batch handling."""

    def __init__(
        self,
        model: Model,
        X_baseline: Tensor,
        ref_point: Tensor,
        *,
        utility_values: Sequence[Sequence[float]] | Sequence[float] | Tensor | None = None,
        objective: MCMultiOutputObjective | None = None,
        weights: Tensor | None = None,
        sampler: MCSampler | None = None,
        ordinal_likelihoods: Sequence[nn.Module] | nn.Module | None = None,
        objective_signs: Sequence[float] | Tensor | None = None,
        train_Y: Tensor | None = None,
        Y_baseline: Tensor | None = None,
        class_offset: int = 0,
        link: str = "auto",
        input_perturbation_n_w: int | None = None,
        risk_type=None,
        risk_alpha: float = 0.5,
        rho: float = 0.05,
        constraints: Sequence[OutcomeConstraint] | None = None,
        eta: float | Tensor = 1e-3,
        fat: bool = False,
    ) -> None:
        utility_values = _with_default_utility_values(model, utility_values)
        if Y_baseline is not None:
            Y_baseline = torch.as_tensor(Y_baseline)
            if not bool(torch.isfinite(Y_baseline).all()):
                Y_baseline = None
        if train_Y is None and Y_baseline is None:
            train_Y = infer_multioutput_ordinal_train_y(model)
        if train_Y is not None:
            train_Y = complete_ordinal_baseline_rows(train_Y)
        objective = _resolve_ordinal_objective(
            model=model,
            utility_values=utility_values,
            objective=objective,
            ordinal_likelihoods=ordinal_likelihoods,
            objective_signs=objective_signs,
            link=link,
            input_perturbation_n_w=input_perturbation_n_w,
            risk_type=risk_type,
            risk_alpha=risk_alpha,
        )
        super().__init__(
            model=model,
            X_baseline=X_baseline,
            ref_point=ref_point,
            utility_values=utility_values,
            objective=objective,
            weights=weights,
            sampler=sampler,
            ordinal_likelihoods=ordinal_likelihoods,
            objective_signs=objective_signs,
            train_Y=train_Y,
            Y_baseline=Y_baseline,
            class_offset=class_offset,
            link=link,
            input_perturbation_n_w=input_perturbation_n_w,
            risk_type=risk_type,
            risk_alpha=risk_alpha,
            rho=rho,
        )
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

    def _evaluate(self, X: Tensor) -> Tensor:
        posterior = self.model.posterior(X)
        samples = self.get_posterior_samples(posterior)
        objective_values = self.base_objective(samples, X=X)
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
        return _multi_output._reduce_sample_and_q_to_tbatch(improvement, X)

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        Xq = _multi_output.ensure_q_batch(X)
        try:
            return self._evaluate(Xq)
        except RuntimeError as err:
            if not any(message in str(err) for message in _NPAREGO_TBATCH_ERRORS):
                raise
            batch_shape = Xq.shape[:-2]
            if len(batch_shape) == 0:
                raise
            q, d = int(Xq.shape[-2]), int(Xq.shape[-1])
            X_flat = Xq.reshape(-1, q, d)
            values = [self._evaluate(X_i) for X_i in X_flat]
            return torch.stack(values).reshape(batch_shape)


class qOrdinalExpectedHypervolumeImprovement(
    qMultiOutputOrdinalExpectedHypervolumeImprovement
):
    """Ordinal multi-objective qEHVI with domain-first naming."""


class qOrdinalNoisyExpectedHypervolumeImprovement(
    qMultiOutputOrdinalNoisyExpectedHypervolumeImprovement
):
    """Ordinal multi-objective qNEHVI with domain-first naming."""


class qOrdinalNParEGO(qMultiOutputOrdinalNParEGO):
    """Ordinal multi-objective NParEGO with domain-first naming."""


class qHeteroOrdinalExpectedHypervolumeImprovement(
    qHeteroMultiOutputOrdinalExpectedHypervolumeImprovement
):
    """Heteroscedastic ordinal qEHVI with domain-first naming."""


class qHeteroOrdinalNoisyExpectedHypervolumeImprovement(
    qHeteroMultiOutputOrdinalNoisyExpectedHypervolumeImprovement
):
    """Heteroscedastic ordinal qNEHVI with domain-first naming."""


class qHeteroOrdinalNParEGO(qHeteroMultiOutputOrdinalNParEGO):
    """Heteroscedastic ordinal NParEGO with domain-first naming."""


__all__ = [
    "compute_observed_ordinal_utility",
    "compute_ordinal_expected_utility_best_f",
    "qHeteroMultiOutputOrdinalExpectedImprovement",
    "qHeteroMultiOutputOrdinalExpectedUtility",
    "qHeteroMultiOutputOrdinalNormalScoreObjective",
    "qHeteroMultiOutputOrdinalProbabilityOfImprovement",
    "qHeteroOrdinalExpectedHypervolumeImprovement",
    "qHeteroOrdinalExpectedImprovement",
    "qHeteroOrdinalExpectedUtility",
    "qHeteroOrdinalExpectedUtilityUpperConfidenceBound",
    "qHeteroOrdinalNParEGO",
    "qHeteroOrdinalNoisyExpectedHypervolumeImprovement",
    "qHeteroOrdinalProbabilityOfImprovement",
    "qMultiOutputOrdinalUtilityObjective",
    "qOrdinalExpectedHypervolumeImprovement",
    "qOrdinalExpectedImprovement",
    "qOrdinalExpectedUtility",
    "qOrdinalKnowledgeGradient",
    "qOrdinalNParEGO",
    "qOrdinalNoisyExpectedHypervolumeImprovement",
    "qOrdinalProbabilityOfFeasibility",
    "qOrdinalProbabilityOfImprovement",
    "qOrdinalUpperConfidenceBound",
]
