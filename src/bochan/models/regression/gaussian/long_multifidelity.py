"""Long-format Gaussian multi-fidelity regression models.

The public input already contains one or more fidelity columns, e.g.
``X = [design..., fidelity]``. This is intentionally separate from the existing
wide-format multi-fidelity models where fidelities are represented by output
columns.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from gpytorch.kernels import Kernel
from gpytorch.likelihoods import Likelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch import Tensor

from bochan.models.components.mixed_kronecker import (
    validate_mixed_input_transform_for_training,
)
from bochan.models.components.mixed_multifidelity import (
    build_mixed_non_fidelity_kernel,
)
from bochan.models.multifidelity import FidelitySpec, ResolvedFidelitySpec


class GaussianMultiFidelityGP(SingleTaskMultiFidelityGP):
    """Gaussian multi-fidelity GP for long-format inputs.

    Parameters
    ----------
    train_X:
        Training inputs with fidelity represented by an input feature.
    train_Y:
        Scalar observations with shape ``[n, 1]``.
    train_Yvar:
        Optional known observation variances with the same shape as ``train_Y``.
    fidelity_spec:
        Shared fidelity-axis contract. Negative indices are resolved against
        ``train_X.shape[-1]``.
    bounds:
        Optional public input bounds used to validate target fidelities.
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor | None = None,
        *,
        fidelity_spec: FidelitySpec | ResolvedFidelitySpec,
        bounds: Tensor | None = None,
        linear_truncated: bool = True,
        nu: float = 2.5,
        covar_module: Kernel | None = None,
        likelihood: Likelihood | None = None,
        outcome_transform: OutcomeTransform | None = None,
        input_transform: InputTransform | None = None,
    ) -> None:
        X = torch.as_tensor(train_X)
        Y = torch.as_tensor(train_Y, dtype=X.dtype, device=X.device)
        if X.ndim != 2:
            raise ValueError("train_X must have shape [n, d].")
        if Y.ndim != 2 or Y.shape != (X.shape[0], 1):
            raise ValueError("train_Y must have shape [n, 1] for scalar regression.")
        if not torch.isfinite(X).all() or not torch.isfinite(Y).all():
            raise ValueError("train_X and train_Y must contain only finite values.")

        Yvar = None
        if train_Yvar is not None:
            Yvar = torch.as_tensor(train_Yvar, dtype=X.dtype, device=X.device)
            if Yvar.shape != Y.shape:
                raise ValueError("train_Yvar must have the same shape as train_Y.")
            if not torch.isfinite(Yvar).all() or bool((Yvar < 0).any()):
                raise ValueError("train_Yvar must be finite and non-negative.")

        d = int(X.shape[-1])
        if isinstance(fidelity_spec, FidelitySpec):
            resolved = fidelity_spec.resolve(d=d, bounds=bounds, single_fidelity_only=True)
        elif isinstance(fidelity_spec, ResolvedFidelitySpec):
            resolved = fidelity_spec
            if len(resolved.fidelity_features) != 1:
                raise ValueError(
                    "Gaussian Multi-Fidelity v1 supports exactly one continuous fidelity feature."
                )
            index = resolved.primary_fidelity_feature
            if index < 0 or index >= d:
                raise ValueError(f"Invalid resolved fidelity dim {index} for input dim {d}.")
        else:
            raise TypeError("fidelity_spec must be FidelitySpec or ResolvedFidelitySpec.")

        fidelity_index = resolved.primary_fidelity_feature
        super().__init__(
            train_X=X,
            train_Y=Y,
            train_Yvar=Yvar,
            data_fidelities=[fidelity_index],
            linear_truncated=linear_truncated,
            nu=nu,
            covar_module=covar_module,
            likelihood=likelihood,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
        )

        self.fidelity_spec = resolved
        self.fidelity_features = tuple(resolved.fidelity_features)
        self.target_fidelities = (
            None
            if resolved.target_fidelities is None
            else {int(k): float(v) for k, v in resolved.target_fidelities.items()}
        )
        self.input_mode = "continuous"
        self.fidelity_mode = "feature"
        self.train_X_raw = X.detach().clone()
        self.train_Y_raw = Y.detach().clone()
        self.train_Yvar_raw = None if Yvar is None else Yvar.detach().clone()

    @property
    def fidelity_metadata(self) -> dict[str, Any]:
        """Return JSON-serializable metadata for persistence and serving."""

        return {
            "fidelity_mode": self.fidelity_mode,
            "fidelity_features": list(self.fidelity_features),
            "target_fidelities": self.target_fidelities,
            "input_mode": self.input_mode,
            "cat_dims": [],
        }

    def make_mll(self) -> ExactMarginalLogLikelihood:
        """Return the exact marginal log likelihood used by standard fit helpers."""

        return ExactMarginalLogLikelihood(self.likelihood, self)

    def predict(
        self,
        X: Tensor,
        *,
        observation_noise: bool | Tensor = False,
        posterior_transform: Any = None,
    ) -> tuple[Tensor, Tensor]:
        """Return posterior mean and standard deviation at public long-format inputs."""

        posterior = self.posterior(
            X,
            observation_noise=observation_noise,
            posterior_transform=posterior_transform,
        )
        return posterior.mean, posterior.variance.clamp_min(0.0).sqrt()


class GaussianMixedMultiFidelityGP(GaussianMultiFidelityGP):
    """Mixed continuous/categorical Gaussian multi-fidelity GP.

    The covariance is constructed as ``K_continuous * K_categorical`` for
    non-fidelity inputs. ``SingleTaskMultiFidelityGP`` then multiplies this by
    its dedicated fidelity kernel, producing
    ``K_continuous * K_categorical * K_fidelity``.
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor | None = None,
        *,
        cat_dims: Sequence[int],
        fidelity_spec: FidelitySpec | ResolvedFidelitySpec,
        bounds: Tensor | None = None,
        covar_module: Kernel | None = None,
        likelihood: Likelihood | None = None,
        outcome_transform: OutcomeTransform | None = None,
        input_transform: InputTransform | None = None,
    ) -> None:
        X = torch.as_tensor(train_X)
        if X.ndim != 2:
            raise ValueError("train_X must have shape [n, d].")
        if len(tuple(cat_dims)) == 0:
            raise ValueError("cat_dims must contain at least one categorical feature index.")

        d = int(X.shape[-1])
        if isinstance(fidelity_spec, FidelitySpec):
            spec = fidelity_spec
        elif isinstance(fidelity_spec, ResolvedFidelitySpec):
            spec = FidelitySpec(
                fidelity_features=tuple(fidelity_spec.fidelity_features),
                target_fidelities=fidelity_spec.target_fidelities,
            )
        else:
            raise TypeError("fidelity_spec must be FidelitySpec or ResolvedFidelitySpec.")

        resolved = spec.resolve(
            d=d,
            cat_dims=cat_dims,
            bounds=bounds,
            single_fidelity_only=True,
        )
        categorical = tuple(resolved.categorical_features)
        fidelity = tuple(resolved.fidelity_features)
        excluded = set(categorical).union(fidelity)
        continuous = tuple(index for index in range(d) if index not in excluded)

        validate_mixed_input_transform_for_training(
            X,
            input_transform,
            cat_dims=categorical,
        )

        data_covar_module = covar_module
        if data_covar_module is None:
            data_covar_module = build_mixed_non_fidelity_kernel(
                d=d,
                cat_dims=categorical,
                fidelity_dims=fidelity,
            )

        super().__init__(
            train_X=X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            fidelity_spec=resolved,
            bounds=bounds,
            linear_truncated=False,
            covar_module=data_covar_module,
            likelihood=likelihood,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
        )

        self.cat_dims = categorical
        self.cont_dims = continuous
        self.input_mode = "mixed"

    @property
    def fidelity_metadata(self) -> dict[str, Any]:
        """Return JSON-serializable mixed multi-fidelity metadata."""

        return {
            "fidelity_mode": self.fidelity_mode,
            "fidelity_features": list(self.fidelity_features),
            "target_fidelities": self.target_fidelities,
            "input_mode": self.input_mode,
            "cat_dims": list(self.cat_dims),
        }


__all__ = ["GaussianMixedMultiFidelityGP", "GaussianMultiFidelityGP"]
