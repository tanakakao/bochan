from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch
from botorch.acquisition.objective import PosteriorTransform
from botorch.posteriors.posterior import Posterior
from torch import Tensor

from .specs import OutputSpec, PosteriorMode
from .task_aware_posterior import (
    HybridPosteriorComponent,
    TaskAwareHybridPosterior,
)

OutputIndex = int | str


_GH_NODES = (
    -3.1909932017815277,
    -2.266580584531843,
    -1.468553289216668,
    -0.7235510187528376,
    0.0,
    0.7235510187528376,
    1.468553289216668,
    2.266580584531843,
    3.1909932017815277,
)
_GH_WEIGHTS = (
    3.960697726326438e-05,
    0.004943624275536947,
    0.08847452739437657,
    0.43265155900255575,
    0.7202352156060509,
    0.43265155900255575,
    0.08847452739437657,
    0.004943624275536947,
    3.960697726326438e-05,
)


def _call_accessor(
    owner: Any,
    model: Any,
    names: Sequence[str],
    X: Tensor,
    **kwargs: Any,
) -> Any:
    """Call an accessor on a wrapper or its inner model."""

    candidates = [model]
    inner = getattr(model, "model", None)
    if inner is not None and inner is not model:
        candidates.append(inner)

    last_error: Exception | None = None
    for candidate in candidates:
        try:
            return owner._call_accessor(candidate, names, X, **kwargs)
        except (AttributeError, TypeError) as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise AttributeError(f"{type(model).__name__} has none of {tuple(names)}.")


def _latent_posterior(
    owner: Any,
    spec: OutputSpec,
    X: Tensor,
    **kwargs: Any,
) -> Posterior:
    return _call_accessor(
        owner,
        spec.model,
        ("latent_posterior",),
        X,
        **kwargs,
    )


def _select_scalar_samples(
    samples: Tensor,
    *,
    X: Tensor,
    output_index: int,
    name: str,
) -> Tensor:
    """Select one scalar output while preserving acquisition sample axes."""

    out = samples

    # A scalar posterior normally adds a final output dimension to X. Require
    # that extra rank before squeezing it: when q == 1, a transformed sample
    # may legitimately end in a singleton candidate axis with no output axis.
    if out.ndim >= X.ndim + 1 and out.shape[-1] == 1:
        return out.squeeze(-1)

    if out.ndim >= 2 and out.shape[-2] == X.shape[-2]:
        if output_index >= out.shape[-1]:
            raise IndexError(
                f"output_index={output_index} is out of bounds for "
                f"{name}.shape={tuple(out.shape)}."
            )
        return out[..., output_index]

    if output_index == 0:
        return out
    raise RuntimeError(
        f"Could not identify an output axis for {name}. "
        f"samples.shape={tuple(samples.shape)}, X.shape={tuple(X.shape)}."
    )


def _objective_samples(samples: Tensor, spec: OutputSpec) -> Tensor:
    if spec.eq_target is not None:
        out = -torch.abs(samples - float(spec.eq_target)) * spec.weight
    else:
        out = samples * spec.sign * spec.weight
    return spec.transform(out) if spec.transform is not None else out


def _gauss_hermite_moments(
    latent_mean: Tensor,
    latent_variance: Tensor,
    transform: Callable[[Tensor], Tensor],
) -> tuple[Tensor, Tensor]:
    """Deterministically approximate scalar-latent transformed moments."""

    nodes = torch.as_tensor(
        _GH_NODES,
        device=latent_mean.device,
        dtype=latent_mean.dtype,
    )
    weights = torch.as_tensor(
        _GH_WEIGHTS,
        device=latent_mean.device,
        dtype=latent_mean.dtype,
    ) / latent_mean.new_tensor(torch.pi).sqrt()

    node_shape = (nodes.numel(),) + (1,) * latent_mean.ndim
    latent_samples = (
        latent_mean.unsqueeze(0)
        + latent_variance.clamp_min(0.0).sqrt().unsqueeze(0)
        * latent_mean.new_tensor(2.0).sqrt()
        * nodes.reshape(node_shape)
    )
    values = transform(latent_samples)
    weight_shape = (weights.numel(),) + (1,) * (values.ndim - 1)
    normalized_weights = weights.reshape(weight_shape)
    mean = (normalized_weights * values).sum(dim=0)
    variance = (
        normalized_weights * (values - mean.unsqueeze(0)).pow(2)
    ).sum(dim=0)
    return mean, variance.clamp_min(0.0)


def _resolve_binary_likelihood(model: Any, output_index: int) -> Any:
    submodels = getattr(model, "models", None)
    if submodels is not None:
        submodels = list(submodels)
        if 0 <= output_index < len(submodels):
            model = submodels[output_index]

    for candidate in (model, getattr(model, "model", None)):
        if candidate is None:
            continue
        likelihood = getattr(candidate, "likelihood", None)
        if likelihood is not None:
            return likelihood
    raise AttributeError(
        f"Binary likelihood was not found for {type(model).__name__}."
    )


def _binary_probabilities(
    model: Any,
    latent_samples: Tensor,
    *,
    output_index: int,
    eps: float = 1e-6,
) -> Tensor:
    likelihood = _resolve_binary_likelihood(model, output_index)
    conditional = likelihood.forward(latent_samples)
    probabilities = getattr(conditional, "probs", None)
    if probabilities is None:
        probabilities = getattr(conditional, "mean", None)
    if probabilities is None or not torch.is_tensor(probabilities):
        raise TypeError(
            f"{type(likelihood).__name__}.forward(...) did not expose Tensor "
            "probs or mean."
        )
    return probabilities.clamp(eps, 1.0 - eps)


def _binary_value_transform(
    owner: Any,
    spec: OutputSpec,
    X: Tensor,
    output_mode: PosteriorMode,
) -> Callable[[Tensor], Tensor]:
    def transform(latent_samples: Tensor) -> Tensor:
        latent = _select_scalar_samples(
            latent_samples,
            X=X,
            output_index=spec.output_index,
            name=f"{spec.name}.latent_samples",
        )
        p1 = _binary_probabilities(
            spec.model,
            latent,
            output_index=spec.output_index,
        )
        if p1.ndim > latent.ndim and p1.shape[-1] == 1:
            p1 = p1.squeeze(-1)
        selected_probability = 1.0 - p1 if spec.positive_class == 0 else p1

        if output_mode == "probability" or spec.utility_values is None:
            value = selected_probability
        else:
            utilities = owner._as_1d(
                spec.utility_values,
                torch.tensor(
                    [0.0, 1.0],
                    device=p1.device,
                    dtype=p1.dtype,
                ),
                f"{spec.name}.utility_values",
            )
            if utilities.numel() != 2:
                raise ValueError("Binary utility_values must have length 2.")
            value = utilities[0] * (1.0 - p1) + utilities[1] * p1

        if output_mode in ("objective", "expected_utility"):
            value = _objective_samples(value, spec)
        return value

    return transform


def _ordinal_value_transform(
    owner: Any,
    spec: OutputSpec,
    X: Tensor,
    output_mode: PosteriorMode,
) -> Callable[[Tensor], Tensor]:
    likelihood = owner._ordinal_likelihood(spec.model)
    cutpoints = owner._ordinal_cutpoints(likelihood)
    utilities = owner._as_1d(
        spec.utility_values,
        torch.arange(
            cutpoints.numel() + 1,
            device=cutpoints.device,
            dtype=cutpoints.dtype,
        ),
        f"{spec.name}.utility_values",
    )

    def transform(latent_samples: Tensor) -> Tensor:
        latent = _select_scalar_samples(
            latent_samples,
            X=X,
            output_index=spec.output_index,
            name=f"{spec.name}.latent_samples",
        )
        class_probs_from_f = getattr(likelihood, "class_probs_from_f", None)
        if callable(class_probs_from_f):
            probabilities = class_probs_from_f(latent)
        else:
            probabilities = owner._ordinal_probs_from_latent(latent, cutpoints)
        local_utilities = utilities.to(
            device=probabilities.device,
            dtype=probabilities.dtype,
        )
        value = (probabilities * local_utilities).sum(dim=-1)
        if output_mode in ("objective", "expected_utility"):
            value = _objective_samples(value, spec)
        return value

    return transform


def _regression_component(
    owner: Any,
    spec: OutputSpec,
    X: Tensor,
    output_mode: PosteriorMode,
    call_kwargs: dict[str, Any],
) -> HybridPosteriorComponent:
    names = (
        ("latent_posterior", "posterior")
        if output_mode == "latent"
        else ("posterior",)
    )
    posterior = _call_accessor(owner, spec.model, names, X, **call_kwargs)
    mean, variance = owner._regression_stats(
        spec,
        X,
        output_mode,
        **call_kwargs,
    )

    def transform(samples: Tensor) -> Tensor:
        values = _select_scalar_samples(
            samples,
            X=X,
            output_index=spec.output_index,
            name=f"{spec.name}.samples",
        )
        if output_mode in ("objective", "expected_utility"):
            values = _objective_samples(values, spec)
        return values

    return HybridPosteriorComponent(
        mean=mean,
        variance=variance,
        posterior=posterior,
        sample_transform=transform,
        name=spec.name,
    )


def _binary_component(
    owner: Any,
    spec: OutputSpec,
    X: Tensor,
    output_mode: PosteriorMode,
    call_kwargs: dict[str, Any],
) -> HybridPosteriorComponent:
    try:
        posterior = _latent_posterior(owner, spec, X, **call_kwargs)
    except (AttributeError, TypeError):
        mean, _ = owner._binary_stats(
            spec,
            X,
            output_mode,
            **call_kwargs,
        )
        return HybridPosteriorComponent(
            mean=mean,
            variance=torch.zeros_like(mean),
            name=spec.name,
        )

    latent_mean, latent_variance = owner._posterior_mean_variance(
        posterior,
        spec.name,
    )
    latent_mean = owner._select_scalar(
        latent_mean,
        X,
        output_index=spec.output_index,
        name=f"{spec.name}.latent_mean",
    )
    latent_variance = owner._select_scalar(
        latent_variance,
        X,
        output_index=spec.output_index,
        name=f"{spec.name}.latent_variance",
    )
    transform = _binary_value_transform(owner, spec, X, output_mode)
    mean, variance = _gauss_hermite_moments(
        latent_mean,
        latent_variance,
        transform,
    )
    return HybridPosteriorComponent(
        mean=mean,
        variance=variance,
        posterior=posterior,
        sample_transform=transform,
        name=spec.name,
    )


def _ordinal_component(
    owner: Any,
    spec: OutputSpec,
    X: Tensor,
    output_mode: PosteriorMode,
    call_kwargs: dict[str, Any],
) -> HybridPosteriorComponent:
    try:
        posterior = _latent_posterior(owner, spec, X, **call_kwargs)
    except (AttributeError, TypeError):
        mean, _ = owner._ordinal_stats(
            spec,
            X,
            output_mode,
            **call_kwargs,
        )
        return HybridPosteriorComponent(
            mean=mean,
            variance=torch.zeros_like(mean),
            name=spec.name,
        )

    latent_mean, latent_variance = owner._posterior_mean_variance(
        posterior,
        spec.name,
    )
    latent_mean = owner._select_scalar(
        latent_mean,
        X,
        output_index=spec.output_index,
        name=f"{spec.name}.latent_mean",
    )
    latent_variance = owner._select_scalar(
        latent_variance,
        X,
        output_index=spec.output_index,
        name=f"{spec.name}.latent_variance",
    )
    transform = _ordinal_value_transform(owner, spec, X, output_mode)
    mean, variance = _gauss_hermite_moments(
        latent_mean,
        latent_variance,
        transform,
    )
    return HybridPosteriorComponent(
        mean=mean,
        variance=variance,
        posterior=posterior,
        sample_transform=transform,
        name=spec.name,
    )


def _multiclass_component(
    owner: Any,
    spec: OutputSpec,
    X: Tensor,
    output_mode: PosteriorMode,
    call_kwargs: dict[str, Any],
) -> HybridPosteriorComponent:
    # Multiclass wrappers do not yet share one latent-posterior contract. Avoid
    # treating categorical label variance as epistemic uncertainty. Dedicated
    # multiclass acquisition classes remain probability-aware; hybrid generic
    # MC acquisition falls back to the predictive mean until a latent adapter
    # is available for that specific model.
    mean, _ = owner._multiclass_stats(
        spec,
        X,
        output_mode,
        **call_kwargs,
    )
    return HybridPosteriorComponent(
        mean=mean,
        variance=torch.zeros_like(mean),
        name=spec.name,
    )


def _build_component(
    owner: Any,
    spec: OutputSpec,
    X: Tensor,
    output_mode: PosteriorMode,
    call_kwargs: dict[str, Any],
) -> HybridPosteriorComponent:
    if spec.task_type == "regression":
        return _regression_component(owner, spec, X, output_mode, call_kwargs)
    if spec.task_type == "binary":
        return _binary_component(owner, spec, X, output_mode, call_kwargs)
    if spec.task_type == "ordinal":
        return _ordinal_component(owner, spec, X, output_mode, call_kwargs)
    if spec.task_type == "multiclass":
        return _multiclass_component(owner, spec, X, output_mode, call_kwargs)
    raise RuntimeError(f"Unsupported task_type={spec.task_type!r}.")


def _task_aware_posterior(
    self: Any,
    X: Tensor,
    output_indices: OutputIndex | Sequence[OutputIndex] | Tensor | None = None,
    observation_noise: bool | Tensor = False,
    posterior_transform: PosteriorTransform | None = None,
    *,
    output_mode: PosteriorMode = "objective",
    **kwargs: Any,
) -> TaskAwareHybridPosterior:
    if output_mode not in {
        "objective",
        "mean",
        "latent",
        "probability",
        "expected_utility",
    }:
        raise ValueError(f"Unknown output_mode={output_mode!r}.")

    # Latent mode is intentionally unchanged: it already describes a Gaussian
    # latent process and does not contain label / realized-utility variance.
    if output_mode == "latent":
        return self._task_aware_original_posterior(
            X=X,
            output_indices=output_indices,
            observation_noise=observation_noise,
            posterior_transform=posterior_transform,
            output_mode=output_mode,
            **kwargs,
        )

    X = self._unwrap_X(X)
    call_kwargs = dict(kwargs)
    call_kwargs.setdefault("observation_noise", observation_noise)
    call_kwargs.setdefault("posterior_transform", None)

    components = [
        _build_component(
            self,
            self.specs[index],
            X,
            output_mode,
            call_kwargs,
        )
        for index in self._normalize_output_indices(output_indices)
    ]
    mean = self._stack([component.mean for component in components], "mean")
    variance = self._stack(
        [component.variance for component in components],
        "variance",
    )
    posterior = TaskAwareHybridPosterior(
        mean=mean,
        variance=variance,
        components=components,
    )
    return posterior_transform(posterior) if posterior_transform is not None else posterior

__all__ = ["_task_aware_posterior"]
