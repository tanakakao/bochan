"""Production validation helpers for material residual Gaussian models."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from .multi_baseline import MultipleBaselineModelListGP
from .residual import ResidualMaterialGPModel


@dataclass(frozen=True)
class ResidualProductionReport:
    """Summary of one residual-model production health check.

    Args:
        model_type: Concrete model class name.
        num_outputs: Number of posterior outputs observed during validation.
        baseline_output_indices: Outputs backed by deterministic pretrained baselines.
        shared_parameter_aliases: Parameter paths that reference the same tensor object.
        posterior_shape: Shape of the posterior mean at the supplied validation inputs.
    """

    model_type: str
    num_outputs: int
    baseline_output_indices: tuple[int, ...]
    shared_parameter_aliases: tuple[tuple[str, ...], ...]
    posterior_shape: tuple[int, ...]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""

        return {
            "model_type": self.model_type,
            "num_outputs": self.num_outputs,
            "baseline_output_indices": list(self.baseline_output_indices),
            "shared_parameter_aliases": [list(paths) for paths in self.shared_parameter_aliases],
            "posterior_shape": list(self.posterior_shape),
        }


def shared_parameter_aliases(module: nn.Module) -> tuple[tuple[str, ...], ...]:
    """Return parameter paths that alias the same underlying parameter object.

    PyTorch normally de-duplicates shared parameters from ``named_parameters``.
    Production serialization checks need the opposite view so accidental duplicate
    ownership of a pretrained encoder can be detected explicitly.
    """

    paths_by_id: dict[int, list[str]] = defaultdict(list)
    try:
        named_parameters = module.named_parameters(remove_duplicate=False)
    except TypeError:  # pragma: no cover - compatibility with older PyTorch.
        named_parameters = module.named_parameters()
    for name, parameter in named_parameters:
        paths_by_id[id(parameter)].append(name)
    aliases = [tuple(paths) for paths in paths_by_id.values() if len(paths) > 1]
    return tuple(sorted(aliases))


def _baseline_output_indices(model: Any) -> tuple[int, ...]:
    if isinstance(model, MultipleBaselineModelListGP):
        return tuple(int(index) for index in model.baseline_plan.baseline_output_indices)
    if isinstance(model, ResidualMaterialGPModel):
        return (0,)
    models = getattr(model, "models", None)
    if models is None:
        return ()
    return tuple(
        index
        for index, submodel in enumerate(models)
        if isinstance(submodel, ResidualMaterialGPModel)
    )


def validate_residual_production_model(
    model: Any,
    X: Tensor,
    *,
    expected_num_outputs: int | None = None,
    require_finite_variance: bool = True,
) -> ResidualProductionReport:
    """Validate posterior and ownership invariants used by production residual GPs.

    Args:
        model: Residual model or ModelList containing residual submodels.
        X: Validation inputs accepted by ``model.posterior``.
        expected_num_outputs: Optional required posterior output count.
        require_finite_variance: Require finite posterior variance as well as mean.

    Returns:
        A compact health report suitable for smoke tests and diagnostics.

    Raises:
        TypeError: If the model does not expose the expected posterior interface.
        ValueError: If shape, output count, or finite-value checks fail.
    """

    if not torch.is_tensor(X) or X.ndim < 2:
        raise ValueError("X must be a Tensor with shape [..., q, d].")
    posterior = model.posterior(X)
    mean = getattr(posterior, "mean", None)
    variance = getattr(posterior, "variance", None)
    if not torch.is_tensor(mean):
        raise TypeError("Residual production posterior must expose Tensor mean.")
    if mean.ndim < 2:
        raise ValueError("Residual production posterior mean must include q and output dimensions.")
    if not torch.isfinite(mean).all():
        raise ValueError("Residual production posterior mean contains non-finite values.")
    if require_finite_variance:
        if not torch.is_tensor(variance):
            raise TypeError("Residual production posterior must expose Tensor variance.")
        if tuple(variance.shape) != tuple(mean.shape):
            raise ValueError("Residual posterior variance shape must match posterior mean shape.")
        if not torch.isfinite(variance).all():
            raise ValueError("Residual production posterior variance contains non-finite values.")

    num_outputs = int(mean.shape[-1])
    if expected_num_outputs is not None and num_outputs != int(expected_num_outputs):
        raise ValueError(
            "Residual production posterior output count mismatch: "
            f"expected {expected_num_outputs}, got {num_outputs}."
        )

    return ResidualProductionReport(
        model_type=model.__class__.__name__,
        num_outputs=num_outputs,
        baseline_output_indices=_baseline_output_indices(model),
        shared_parameter_aliases=shared_parameter_aliases(model),
        posterior_shape=tuple(int(value) for value in mean.shape),
    )


def assert_residual_posterior_equivalent(
    expected_model: Any,
    actual_model: Any,
    X: Tensor,
    *,
    atol: float = 1e-8,
    rtol: float = 1e-6,
) -> None:
    """Assert that serialization or transport preserved residual posterior outputs."""

    expected = expected_model.posterior(X)
    actual = actual_model.posterior(X)
    torch.testing.assert_close(actual.mean, expected.mean, atol=atol, rtol=rtol)
    torch.testing.assert_close(actual.variance, expected.variance, atol=atol, rtol=rtol)


__all__ = [
    "ResidualProductionReport",
    "assert_residual_posterior_equivalent",
    "shared_parameter_aliases",
    "validate_residual_production_model",
]
