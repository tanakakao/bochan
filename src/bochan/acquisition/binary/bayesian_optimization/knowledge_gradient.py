"""Likelihood-aware one-step Knowledge Gradient for binary classification.

The standard BoTorch qKnowledgeGradient assumes a fantasy-capable regression
posterior. Binary classification observations are Bernoulli labels instead. This
module therefore performs the one-step Bayesian update directly on coherent
posterior function samples: each hypothetical label reweights the function
samples by its Bernoulli likelihood, and the terminal decision value is the best
posterior mean target-class probability on a finite decision set.
"""

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
from bochan.acquisition.binary._probability import (
    latent_samples_to_binary_probabilities,
)
from bochan.acquisition.binary.epistemic import get_binary_latent_posterior


def _coerce_bounds(bounds: Any, *, reference: Tensor) -> Tensor:
    value = torch.as_tensor(bounds, dtype=reference.dtype, device=reference.device)
    if value.ndim != 2 or value.shape[0] != 2:
        raise ValueError(
            "bounds must have shape [2, d] for automatic Binary KG terminal-set "
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
            "Binary KG requires terminal_set, X_baseline, or bounds to define the "
            "finite terminal decision problem."
        )
    reference = torch.as_tensor(reference)
    if reference.ndim == 1:
        reference = reference.unsqueeze(0)
    if reference.ndim != 2:
        raise ValueError("Binary KG terminal inputs must be two-dimensional [n, d].")

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
            "Automatic Binary KG terminal-set generation is continuous-only. "
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


def _align_probability_samples(
    values: Tensor,
    *,
    joint_X: Tensor,
    sample_shape: torch.Size,
) -> Tensor:
    """Return probability samples as [S, *batch, q_total]."""

    sample_ndim = len(sample_shape)
    target_ndim = sample_ndim + joint_X.ndim - 1
    out = values
    while out.ndim > target_ndim and out.shape[-1] == 1:
        out = out.squeeze(-1)
    out = _flatten_sample_shape(out, sample_shape)
    expected_tail = tuple(joint_X.shape[:-1])
    if tuple(out.shape[1:]) != expected_tail:
        raise RuntimeError(
            "Binary KG could not align posterior probability samples with the "
            f"joint decision inputs. samples={tuple(out.shape)}, expected tail={expected_tail}."
        )
    return out


class qBinaryKnowledgeGradient(AcquisitionFunction):
    """One-step KG for maximizing a binary target-class probability.

    The acquisition is a finite-decision-set sample-average approximation (SAA).
    For every candidate ``x`` it draws coherent latent function samples jointly
    at ``x`` and the terminal decision set. Hypothetical Bernoulli labels reweight
    those samples by their exact likelihood. The terminal value is the maximum
    updated posterior mean probability of ``target_class`` on the fixed terminal
    decision set.

    This initial implementation intentionally supports ``q=1``. Non-empty
    ``X_pending`` is rejected because exact pending-aware KG requires nested
    marginalization over the unknown pending labels.
    """

    def __init__(
        self,
        model: Any,
        *,
        terminal_set: Tensor | None = None,
        bounds: Tensor | None = None,
        X_baseline: Tensor | None = None,
        mc_points: Tensor | None = None,
        X_pending: Tensor | None = None,
        target_class: int = 1,
        terminal_size: int = 128,
        num_samples: int = 64,
        sampler: MCSampler | None = None,
        seed: int = 0,
        eps: float = 1e-8,
    ) -> None:
        model = unwrap_single_output_model(model)
        super().__init__(model=model)
        if int(getattr(model, "num_outputs", 1)) != 1:
            raise ValueError("qBinaryKnowledgeGradient currently supports one binary output only.")
        if int(target_class) not in {0, 1}:
            raise ValueError("target_class must be 0 or 1.")
        if num_samples <= 0:
            raise ValueError("num_samples must be positive.")
        if eps <= 0.0:
            raise ValueError("eps must be positive.")

        # mc_points is an existing high-level DataContext field. Treat an
        # explicitly supplied decision grid as the terminal set when terminal_set
        # itself is omitted. The high-level default mc_points=train_X is harmless;
        # bounds, when supplied, still takes precedence for automatic coverage.
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
        self.target_class = int(target_class)
        self.num_samples = int(num_samples)
        self.seed = int(seed)
        self.eps = float(eps)
        self.sampler = sampler
        self.set_X_pending(X_pending)

    def set_X_pending(self, X_pending: Tensor | None = None) -> None:
        if X_pending is not None and torch.as_tensor(X_pending).numel() > 0:
            raise NotImplementedError(
                "qBinaryKnowledgeGradient v1 does not approximate pending-label "
                "conditioning. Generate one candidate at a time, observe its label, "
                "then refit before requesting the next KG candidate."
            )
        self.X_pending = None

    def _joint_inputs(self, X: Tensor) -> Tensor:
        batch_shape = X.shape[:-2]
        terminal = self.terminal_set.to(X)
        view = terminal.view(*([1] * len(batch_shape)), *terminal.shape)
        terminal = view.expand(*batch_shape, *terminal.shape)
        return torch.cat([X, terminal], dim=-2)

    def _probability_samples(self, joint_X: Tensor) -> Tensor:
        posterior = get_binary_latent_posterior(self.model, joint_X)
        if self.sampler is None:
            self.sampler = get_sampler(
                posterior=posterior,
                sample_shape=torch.Size([self.num_samples]),
                seed=self.seed,
            )
        latent_samples = self.sampler(posterior)
        probabilities = latent_samples_to_binary_probabilities(
            self.model,
            latent_samples,
            eps=self.eps,
            name="Binary KG latent samples",
        )
        return _align_probability_samples(
            probabilities,
            joint_X=joint_X,
            sample_shape=self.sampler.sample_shape,
        )

    def _terminal_value_after_outcome(
        self,
        *,
        decision_samples: Tensor,
        weights: Tensor,
    ) -> Tensor:
        denominator = weights.sum(dim=0).clamp_min(self.eps)
        updated_mean = (
            weights.unsqueeze(-1) * decision_samples
        ).sum(dim=0) / denominator.unsqueeze(-1)
        return updated_mean.max(dim=-1).values

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        joint_X = self._joint_inputs(X)
        p_one = self._probability_samples(joint_X)
        p_at_candidate = p_one[..., 0]
        terminal_p_one = p_one[..., 1:]
        decision_samples = (
            terminal_p_one if self.target_class == 1 else 1.0 - terminal_p_one
        )

        current_value = decision_samples.mean(dim=0).max(dim=-1).values
        weight_one = p_at_candidate.clamp(self.eps, 1.0 - self.eps)
        weight_zero = (1.0 - p_at_candidate).clamp(self.eps, 1.0 - self.eps)
        value_one = self._terminal_value_after_outcome(
            decision_samples=decision_samples,
            weights=weight_one,
        )
        value_zero = self._terminal_value_after_outcome(
            decision_samples=decision_samples,
            weights=weight_zero,
        )

        predictive_one = weight_one.mean(dim=0)
        predictive_zero = weight_zero.mean(dim=0)
        expected_terminal_value = predictive_one * value_one + predictive_zero * value_zero
        return (expected_terminal_value - current_value).clamp_min(0.0)


__all__ = ["qBinaryKnowledgeGradient"]
