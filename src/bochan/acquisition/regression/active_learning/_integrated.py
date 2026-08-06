"""Integrated-posterior-variance regression acquisitions."""

from __future__ import annotations

from typing import Any

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor

from ._base import _RegressionActiveLearningBase
from ._base_common import _ensure_q_batch

try:
    from botorch.acquisition.active_learning import (
        qNegIntegratedPosteriorVariance as _BoTorchQNegIntegratedPosteriorVariance,
    )
except Exception:  # pragma: no cover - depends on BoTorch version
    _BoTorchQNegIntegratedPosteriorVariance = None


class qRegressionNegIntegratedPosteriorVariance(AcquisitionFunction):
    """True BoTorch qNegIntegratedPosteriorVariance wrapper.

    This delegates to BoTorch's implementation and therefore requires a model
    that supports the operations expected by BoTorch, especially fantasize().
    Use qRegressionIntegratedPosteriorVarianceProxy for DeepGP / custom models
    that do not support fantasize().
    """

    def __init__(
        self,
        model,
        mc_points: Tensor,
        *,
        sampler: Any | None = None,
        objective: Any | None = None,
        posterior_transform: Any | None = None,
        X_pending: Tensor | None = None,
        **kwargs: Any,
    ) -> None:
        if _BoTorchQNegIntegratedPosteriorVariance is None:
            raise ImportError(
                "botorch.acquisition.active_learning.qNegIntegratedPosteriorVariance "
                "is not available in this BoTorch version."
            )

        super().__init__(model=model)

        init_kwargs: dict[str, Any] = {
            "model": model,
            "mc_points": mc_points,
        }
        if sampler is not None:
            init_kwargs["sampler"] = sampler
        if objective is not None:
            init_kwargs["objective"] = objective
        if posterior_transform is not None:
            init_kwargs["posterior_transform"] = posterior_transform
        if X_pending is not None:
            init_kwargs["X_pending"] = X_pending
        init_kwargs.update(kwargs)

        # BoTorch signatures differ slightly across versions.  Try the most
        # complete call first, then progressively remove optional keywords.
        try:
            self.acqf = _BoTorchQNegIntegratedPosteriorVariance(**init_kwargs)
        except TypeError:
            for key in ("X_pending", "posterior_transform", "objective", "sampler"):
                init_kwargs.pop(key, None)
                try:
                    self.acqf = _BoTorchQNegIntegratedPosteriorVariance(**init_kwargs)
                    break
                except TypeError:
                    continue
            else:
                raise

    def set_X_pending(self, X_pending: Tensor | None = None) -> None:
        if hasattr(self.acqf, "set_X_pending"):
            self.acqf.set_X_pending(X_pending)
        else:
            self.acqf.X_pending = X_pending

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        return self.acqf(X)


class qRegressionIntegratedPosteriorVarianceProxy(_RegressionActiveLearningBase):
    """Lightweight integrated-posterior-variance proxy.

    This is not BoTorch qNegIntegratedPosteriorVariance.  It does not fantasize.
    It scores candidates by how much they cover high-variance reference regions.
    """

    def __init__(
        self,
        model,
        X_ref: Tensor,
        *,
        kernel_lengthscale: float = 0.2,
        normalize_weights: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, **kwargs)
        if X_ref.ndim != 2:
            raise ValueError(f"X_ref must have shape [n_ref, d]. Got {tuple(X_ref.shape)}.")
        self.register_buffer("X_ref", X_ref.detach().clone())
        self.kernel_lengthscale = float(kernel_lengthscale)
        self.normalize_weights = bool(normalize_weights)

    def _reference_variance(self) -> Tensor:
        _, ref_var, _ = self._posterior_mean_variance(self.X_ref, observation_noise=False)
        n_ref = int(self.X_ref.shape[-2])
        ref_var = self._aggregate_n_w_if_needed(
            ref_var,
            q=n_ref,
            context="qRegressionIntegratedPosteriorVarianceProxy reference variance",
        )
        if ref_var.shape[-1] != n_ref:
            raise RuntimeError(
                "Reference variance must have last dimension n_ref. "
                f"ref_var.shape={tuple(ref_var.shape)}, n_ref={n_ref}."
            )
        return ref_var

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        raw_X = _ensure_q_batch(X)
        Xt = self._apply_input_transform_for_distance(raw_X)

        ref_var = self._reference_variance()
        X_ref_t = self._reference_to_distance_space(self.X_ref, like=Xt)
        if X_ref_t is None:
            raise RuntimeError("X_ref unexpectedly became None after transform.")
        X_ref_2d = X_ref_t.reshape(-1, X_ref_t.shape[-1])

        if ref_var.ndim > 1:
            # If reference variance has extra leading dimensions, average them.
            while ref_var.ndim > 1:
                ref_var = ref_var.mean(dim=0)

        if ref_var.shape[-1] != X_ref_2d.shape[-2]:
            # InputPerturbation may expand X_ref in distance space.  Collapse
            # repeated reference points back to nominal reference count if possible.
            n_ref = int(self.X_ref.shape[-2])
            if X_ref_2d.shape[-2] % n_ref == 0:
                n_w_ref = X_ref_2d.shape[-2] // n_ref
                X_ref_2d = X_ref_2d.reshape(n_ref, n_w_ref, X_ref_2d.shape[-1]).mean(dim=1)
            if ref_var.shape[-1] != X_ref_2d.shape[-2]:
                raise RuntimeError(
                    "Reference variance / reference point mismatch. "
                    f"ref_var.shape={tuple(ref_var.shape)}, X_ref_2d.shape={tuple(X_ref_2d.shape)}."
                )

        d2 = torch.cdist(Xt.reshape(-1, Xt.shape[-1]), X_ref_2d).pow(2)
        d2 = d2.reshape(*Xt.shape[:-1], X_ref_2d.shape[-2])

        ls2 = max(self.kernel_lengthscale ** 2, self.eps)
        weights = torch.exp(-0.5 * d2 / ls2)
        if self.normalize_weights:
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(self.eps)

        view_shape = (1,) * (weights.ndim - 1) + (ref_var.shape[-1],)
        score = (weights * ref_var.view(*view_shape)).sum(dim=-1)

        return self._finalize_pointwise_score(
            score,
            raw_X,
            Xt,
            name="qRegressionIntegratedPosteriorVarianceProxy",
        )
