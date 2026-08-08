"""Likelihood-aware one-step Knowledge Gradient for ordinal outcomes."""

from __future__ import annotations

import math
from typing import Any

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.sampling.base import MCSampler
from botorch.sampling.get_sampler import get_sampler
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor

from bochan.acquisition._duplicate_exclusion import unwrap_single_output_model
from bochan.acquisition.ordinal.active_learning.single_output import (
    _resolve_ordinal_likelihood,
)


def _coerce_bounds(bounds: Any, *, reference: Tensor) -> Tensor:
    value = torch.as_tensor(bounds, dtype=reference.dtype, device=reference.device)
    if value.ndim != 2 or value.shape[0] != 2:
        raise ValueError(
            "bounds must have shape [2, d] for automatic Ordinal KG terminal-set "
            f"generation. Got shape={tuple(value.shape)}."
        )
    if value.shape[-1] != reference.shape[-1]:
        raise ValueError(
            "bounds and X_baseline must use the same input dimension. "
            f"Got bounds d={value.shape[-1]} and X_baseline d={reference.shape[-1]}."
        )
    if not torch.isfinite(value).all() or torch.any(value[1] <= value[0]):
        raise ValueError("bounds must be finite and every upper bound must exceed the lower bound.")
    return value


def _coerce_points(value: Any, *, reference: Tensor, name: str) -> Tensor:
    points = torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
    if points.ndim == 1:
        points = points.unsqueeze(0)
    if points.ndim != 2 or points.shape[-1] != reference.shape[-1]:
        raise ValueError(
            f"{name} must have shape [n, d] with d={reference.shape[-1]}. "
            f"Got shape={tuple(points.shape)}."
        )
    if points.shape[0] == 0 or not torch.isfinite(points).all():
        raise ValueError(f"{name} must contain at least one finite point.")
    return points


def _model_cat_dims(model: Any) -> list[int]:
    for owner in (model, getattr(model, "model", None)):
        if owner is None:
            continue
        dims = getattr(owner, "cat_dims", None)
        if dims is not None:
            return [int(index) for index in dims]
    return []


def _build_terminal_set(
    *,
    model: Any,
    X_baseline: Tensor | None,
    terminal_set: Tensor | None,
    bounds: Tensor | None,
    terminal_size: int,
    seed: int,
) -> Tensor:
    reference = X_baseline
    if reference is None:
        reference = terminal_set
    if reference is None:
        reference = bounds
    if reference is None:
        raise ValueError(
            "Ordinal KG requires terminal_set, X_baseline, or bounds to define "
            "the finite terminal decision problem."
        )
    reference = torch.as_tensor(reference)
    if reference.ndim == 1:
        reference = reference.unsqueeze(0)
    if reference.ndim != 2:
        raise ValueError("Ordinal KG terminal inputs must be two-dimensional [n, d].")

    if terminal_set is not None:
        return _coerce_points(terminal_set, reference=reference, name="terminal_set")

    baseline = None
    if X_baseline is not None:
        baseline = _coerce_points(X_baseline, reference=reference, name="X_baseline")

    if bounds is None:
        if baseline is None:
            raise ValueError("bounds is required when terminal_set and X_baseline are absent.")
        return baseline

    if _model_cat_dims(model):
        raise ValueError(
            "Automatic Ordinal KG terminal-set generation is continuous-only. "
            "For mixed/categorical inputs pass an explicit terminal_set containing "
            "valid category assignments."
        )
    if terminal_size <= 0:
        raise ValueError("terminal_size must be positive.")

    bounds_tensor = _coerce_bounds(bounds, reference=reference)
    sobol = draw_sobol_samples(
        bounds=bounds_tensor,
        n=int(terminal_size),
        q=1,
        seed=int(seed),
    ).squeeze(-2)
    if baseline is not None:
        sobol = torch.cat([baseline.to(sobol), sobol], dim=0)
    return sobol


def _flatten_sample_shape(values: Tensor, sample_shape: torch.Size) -> Tensor:
    sample_ndim = len(sample_shape)
    if sample_ndim == 0:
        return values.unsqueeze(0)
    expected = math.prod(int(size) for size in sample_shape)
    return values.reshape(expected, *values.shape[sample_ndim:])


def _latent_posterior(model: Any, X: Tensor):
    accessor = getattr(model, "latent_posterior", None)
    if callable(accessor):
        return accessor(X)
    return model.posterior(X)


def _align_latent_samples(
    values: Tensor,
    *,
    joint_X: Tensor,
    sample_shape: torch.Size,
) -> Tensor:
    sample_ndim = len(sample_shape)
    target_ndim = sample_ndim + joint_X.ndim - 1
    out = values
    while out.ndim > target_ndim and out.shape[-1] == 1:
        out = out.squeeze(-1)
    out = _flatten_sample_shape(out, sample_shape)
    expected_tail = tuple(joint_X.shape[:-1])
    if tuple(out.shape[1:]) != expected_tail:
        raise RuntimeError(
            "Ordinal KG could not align latent posterior samples with the joint "
            f"decision inputs. samples={tuple(out.shape)}, expected tail={expected_tail}."
        )
    return out


def _resolve_utility_values(
    *,
    class_probs: Tensor,
    utility_values: Tensor | None,
    objective: Any | None,
) -> Tensor:
    values = utility_values
    if values is None and objective is not None:
        values = getattr(objective, "utility_values", None)
        if values is None:
            raise ValueError(
                "Ordinal KG objective must expose utility_values. Prefer "
                "OrdinalExpectedUtilityMCObjective or pass utility_values directly."
            )
    if values is None:
        values = torch.arange(
            class_probs.shape[-1],
            dtype=class_probs.dtype,
            device=class_probs.device,
        )
    else:
        values = torch.as_tensor(
            values,
            dtype=class_probs.dtype,
            device=class_probs.device,
        ).reshape(-1)
    if values.numel() != class_probs.shape[-1]:
        raise ValueError(
            "utility_values length must equal the number of ordinal classes. "
            f"Got {values.numel()} utilities for {class_probs.shape[-1]} classes."
        )
    if not torch.isfinite(values).all():
        raise ValueError("utility_values must be finite.")
    return values


class qOrdinalKnowledgeGradient(AcquisitionFunction):
    """One-step KG for maximizing expected ordinal utility.

    Hypothetical ordinal labels reweight coherent latent function samples using
    ``ordinal_likelihood.class_probs_from_f``. For each possible observed class,
    the acquisition computes the best updated posterior mean expected utility on
    the finite terminal decision set and averages these terminal values under the
    predictive class probabilities.

    ``q=1`` is intentional in this first implementation. Pending labels are not
    approximated as known outcomes.
    """

    def __init__(
        self,
        model: Any,
        *,
        ordinal_likelihood: Any | None = None,
        utility_values: Tensor | None = None,
        objective: Any | None = None,
        terminal_set: Tensor | None = None,
        bounds: Tensor | None = None,
        X_baseline: Tensor | None = None,
        mc_points: Tensor | None = None,
        X_pending: Tensor | None = None,
        terminal_size: int = 128,
        num_samples: int = 64,
        sampler: MCSampler | None = None,
        seed: int = 0,
        eps: float = 1e-8,
    ) -> None:
        model = unwrap_single_output_model(model)
        super().__init__(model=model)
        if int(getattr(model, "num_outputs", 1)) != 1:
            raise ValueError("qOrdinalKnowledgeGradient currently supports one ordinal output only.")
        if num_samples <= 0:
            raise ValueError("num_samples must be positive.")
        if eps <= 0.0:
            raise ValueError("eps must be positive.")

        self.ordinal_likelihood = _resolve_ordinal_likelihood(
            model=model,
            ordinal_likelihood=ordinal_likelihood,
        )
        self.utility_values = utility_values
        self.objective = objective

        if terminal_set is None and bounds is None and mc_points is not None:
            terminal_set = mc_points
        resolved_terminal = _build_terminal_set(
            model=model,
            X_baseline=X_baseline,
            terminal_set=terminal_set,
            bounds=bounds,
            terminal_size=int(terminal_size),
            seed=int(seed),
        )
        self.register_buffer("terminal_set", resolved_terminal.detach().clone())
        self.num_samples = int(num_samples)
        self.seed = int(seed)
        self.eps = float(eps)
        self.sampler = sampler
        self.set_X_pending(X_pending)

    def set_X_pending(self, X_pending: Tensor | None = None) -> None:
        if X_pending is not None and torch.as_tensor(X_pending).numel() > 0:
            raise NotImplementedError(
                "qOrdinalKnowledgeGradient v1 does not approximate pending-label "
                "conditioning. Generate one candidate at a time, observe its class, "
                "then refit before requesting the next KG candidate."
            )
        self.X_pending = None

    def _joint_inputs(self, X: Tensor) -> Tensor:
        batch_shape = X.shape[:-2]
        terminal = self.terminal_set.to(X)
        view = terminal.view(*([1] * len(batch_shape)), *terminal.shape)
        terminal = view.expand(*batch_shape, *terminal.shape)
        return torch.cat([X, terminal], dim=-2)

    def _class_probability_samples(self, joint_X: Tensor) -> Tensor:
        posterior = _latent_posterior(self.model, joint_X)
        if self.sampler is None:
            self.sampler = get_sampler(
                posterior=posterior,
                sample_shape=torch.Size([self.num_samples]),
                seed=self.seed,
            )
        latent = self.sampler(posterior)
        latent = _align_latent_samples(
            latent,
            joint_X=joint_X,
            sample_shape=self.sampler.sample_shape,
        )
        probs = self.ordinal_likelihood.class_probs_from_f(latent)
        probs = probs.clamp_min(self.eps)
        return probs / probs.sum(dim=-1, keepdim=True).clamp_min(self.eps)

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        joint_X = self._joint_inputs(X)
        class_probs = self._class_probability_samples(joint_X)
        utilities = _resolve_utility_values(
            class_probs=class_probs,
            utility_values=self.utility_values,
            objective=self.objective,
        )
        decision_samples = (class_probs * utilities).sum(dim=-1)
        current_value = decision_samples.mean(dim=0).max(dim=-1).values

        candidate_weights = class_probs[..., 0, :]
        denominator = candidate_weights.sum(dim=0).clamp_min(self.eps)
        weighted_terminal = (
            candidate_weights.unsqueeze(-1) * decision_samples.unsqueeze(-2)
        ).sum(dim=0) / denominator.unsqueeze(-1)
        value_after_class = weighted_terminal.max(dim=-1).values
        predictive_class_probs = candidate_weights.mean(dim=0)
        expected_terminal_value = (
            predictive_class_probs * value_after_class
        ).sum(dim=-1)
        return (expected_terminal_value - current_value).clamp_min(0.0)


__all__ = ["qOrdinalKnowledgeGradient"]
