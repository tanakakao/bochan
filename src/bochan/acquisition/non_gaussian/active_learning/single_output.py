"""Active learning for non-Gaussian response posteriors."""
from __future__ import annotations

import warnings
from typing import Any, Literal

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.sampling.base import MCSampler
from botorch.sampling.get_sampler import get_sampler
from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor

from bochan.acquisition.regression.active_learning import (
    qRegressionNegIntegratedPosteriorVariance,
)
from bochan.acquisition.regression.active_learning.single_output import (
    _RegressionActiveLearningBase,
)

from .._stats import ensure_q_batch, non_gaussian_response_stats, safe_logdet

_DEFAULT_SAMPLE_SHAPE = torch.Size([128])


class _NonGaussianActiveLearningBase(_RegressionActiveLearningBase):
    """Common fixed-base-sample response-statistics acquisition base."""

    def __init__(
        self,
        model,
        *,
        sampler: MCSampler | None = None,
        sample_shape: torch.Size = _DEFAULT_SAMPLE_SHAPE,
        seed: int | None = None,
        num_samples: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, **kwargs)
        self.sampler = sampler
        self.sample_shape = (
            torch.Size([num_samples]) if num_samples is not None else sample_shape
        )
        self.seed = seed

    def _stats(self, X: Tensor):
        """Return response-scale statistics using persistent base samples."""
        Xq = ensure_q_batch(X)
        if self.sampler is None and not getattr(
            self.model, "is_non_gaussian_model_list", False
        ):
            posterior = self.model.posterior(Xq, observation_noise=False)
            self.sampler = get_sampler(
                posterior,
                self.sample_shape,
                seed=self.seed,
            )
        return non_gaussian_response_stats(
            self.model,
            Xq,
            sampler=self.sampler,
            sample_shape=self.sample_shape,
            seed=self.seed,
            eps=self.eps,
        )

    def _pointwise(self, X: Tensor, field: str) -> Tensor:
        """Finalize a pointwise response statistic with common penalties."""
        Xq = ensure_q_batch(X)
        value = getattr(self._stats(Xq), field)
        Xt = self._apply_input_transform_for_distance(Xq)
        return self._finalize_pointwise_score(
            value,
            X,
            Xt,
            name=type(self).__name__,
        )

    def _finalize_joint_score(self, score: Tensor, X: Tensor) -> Tensor:
        """Apply duplicate / pending / observed penalties to a joint score."""
        Xq = ensure_q_batch(X)
        Xt = self._apply_input_transform_for_distance(Xq)
        penalty = self._reduce_q(self._total_penalty_per_point(Xt))
        result = score - penalty
        target = torch.Size(Xq.shape[:-2])
        if result.shape == target:
            return result
        while result.ndim > len(target):
            result = result.mean(dim=0)
        if result.shape == target:
            return result
        target_numel = 1
        for size in target:
            target_numel *= int(size)
        if result.numel() == target_numel:
            return result.reshape(target)
        raise RuntimeError(
            f"{type(self).__name__}: joint score shape mismatch. "
            f"score.shape={tuple(score.shape)}, expected={tuple(target)}."
        )


def _field_class(name: str, field: str, doc: str):
    """Create a documented pointwise acquisition with a shared implementation."""

    def forward(self, X: Tensor) -> Tensor:
        """Evaluate the response statistic at raw-space candidates."""
        return self._pointwise(X, field)

    return type(
        name,
        (_NonGaussianActiveLearningBase,),
        {
            "__doc__": doc,
            "forward": t_batch_mode_transform()(forward),
        },
    )


qNonGaussianResponseMeanVariance = _field_class(
    "qNonGaussianResponseMeanVariance",
    "response_mean_variance",
    """Evaluate ``Var_f[E(Y|f)]``, the standard non-Gaussian AL score.""",
)
qNonGaussianExpectedObservationVariance = _field_class(
    "qNonGaussianExpectedObservationVariance",
    "expected_observation_variance",
    """Evaluate aleatoric variance for noise diagnostics.""",
)
qNonGaussianTotalObservationVariance = _field_class(
    "qNonGaussianTotalObservationVariance",
    "total_observation_variance",
    """Evaluate epistemic plus expected observation variance.""",
)
qNonGaussianExpectedObservationEntropy = _field_class(
    "qNonGaussianExpectedObservationEntropy",
    "expected_observation_entropy",
    """Evaluate ``E_f[H(Y|f)]`` with an identified entropy proxy fallback.""",
)
qNonGaussianPredictiveEntropyProxy = _field_class(
    "qNonGaussianPredictiveEntropyProxy",
    "predictive_entropy_proxy",
    """Moment-matched predictive entropy based on total observation variance.""",
)


class qNonGaussianBALDProxy(_NonGaussianActiveLearningBase):
    """Moment proxy for BALD using entropy difference or variance ratio."""

    def __init__(
        self,
        model,
        *,
        method: Literal["variance_ratio", "entropy_difference"] = "entropy_difference",
        **kwargs: Any,
    ) -> None:
        if method not in ("variance_ratio", "entropy_difference"):
            raise ValueError(
                "method must be 'variance_ratio' or 'entropy_difference'."
            )
        super().__init__(model, **kwargs)
        self.method = method

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        """Evaluate the selected non-negative BALD moment proxy."""
        field = (
            "bald_variance_ratio_proxy"
            if self.method == "variance_ratio"
            else "bald_entropy_difference_proxy"
        )
        return self._pointwise(X, field)


def _validate_output_vector(
    value: Tensor | None,
    *,
    num_outputs: int,
    like: Tensor,
    name: str,
) -> Tensor | None:
    """Validate and cast an optional output vector."""
    if value is None:
        return None
    out = value.to(like)
    if out.ndim != 1 or out.numel() != num_outputs:
        raise ValueError(
            f"{name} must be one-dimensional with length {num_outputs}. "
            f"Got shape={tuple(out.shape)}."
        )
    return out


class qNonGaussianIntegratedResponseMeanVarianceProxy(
    _NonGaussianActiveLearningBase
):
    """Candidate-dependent integrated response-mean variance-reduction proxy.

    The score uses the sample covariance between reference response means and
    candidate response means. For reference response vector ``r`` and candidate
    response vector ``c``, the approximate reduction is

    ``diag(C_rc (C_cc + D_obs)^-1 C_cr)``.

    This is a differentiable Gaussian-conditioning moment proxy. It does not
    construct a fantasy model, but unlike the previous implementation it depends
    on the candidate locations and represents an actual variance reduction.
    """

    def __init__(
        self,
        model,
        *,
        mc_points: Tensor | None = None,
        X_ref: Tensor | None = None,
        integration_reduction: Literal["mean", "sum"] = "mean",
        jitter: float = 1e-6,
        **kwargs: Any,
    ) -> None:
        if mc_points is None:
            mc_points = X_ref
        elif X_ref is not None:
            raise ValueError("Specify only one of mc_points and X_ref.")
        if mc_points is None:
            raise ValueError("mc_points is required.")
        if mc_points.ndim != 2:
            raise ValueError(
                "mc_points must have shape [n_ref, d]. "
                f"Got {tuple(mc_points.shape)}."
            )
        if integration_reduction not in ("mean", "sum"):
            raise ValueError(
                "integration_reduction must be 'mean' or 'sum'."
            )
        if jitter <= 0:
            raise ValueError("jitter must be positive.")
        super().__init__(model, **kwargs)
        self.register_buffer("mc_points", mc_points.detach().clone())
        self.integration_reduction = integration_reduction
        self.jitter = float(jitter)

    def _output_reduction(self, reduction: Tensor) -> Tensor:
        """Reduce a ``batch x n_ref x m`` variance-reduction tensor."""
        if self.integration_reduction == "mean":
            value = reduction.mean(dim=-2)
        else:
            value = reduction.sum(dim=-2)

        num_outputs = value.shape[-1]
        scales = _validate_output_vector(
            getattr(self, "output_scales", None),
            num_outputs=num_outputs,
            like=value,
            name="output_scales",
        )
        if scales is not None:
            if torch.any(scales <= 0):
                raise ValueError("output_scales must be positive.")
            value = value / scales.square()

        reduction_name = getattr(self, "multi_output_reduction", None)
        if reduction_name is None:
            return value.squeeze(-1) if num_outputs == 1 else value.mean(dim=-1)
        if reduction_name == "weighted_mean":
            weights = _validate_output_vector(
                getattr(self, "output_weights", None),
                num_outputs=num_outputs,
                like=value,
                name="output_weights",
            )
            if weights is None:
                raise ValueError(
                    "output_weights is required for weighted_mean."
                )
            if torch.any(weights < 0) or weights.sum() <= 0:
                raise ValueError(
                    "output_weights must be non-negative and non-zero."
                )
            return (value * (weights / weights.sum())).sum(dim=-1)
        if reduction_name == "mean":
            return value.mean(dim=-1)
        if reduction_name == "sum":
            return value.sum(dim=-1)
        if reduction_name == "max":
            return value.max(dim=-1).values
        if reduction_name == "min":
            return value.min(dim=-1).values
        raise ValueError(
            f"Unsupported output_reduction={reduction_name!r}."
        )

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        """Evaluate integrated candidate-induced response-mean variance reduction."""
        Xq = ensure_q_batch(X)
        n_ref = int(self.mc_points.shape[-2])
        batch_shape = Xq.shape[:-2]
        refs = self.mc_points.to(Xq)
        refs = refs.reshape(
            *((1,) * len(batch_shape)),
            n_ref,
            refs.shape[-1],
        ).expand(*batch_shape, n_ref, refs.shape[-1])
        combined = torch.cat([refs, Xq], dim=-2)
        stats = self._stats(combined)
        samples = stats.response_mean_samples
        ref_samples = samples[..., :n_ref, :]
        candidate_samples = samples[..., n_ref:, :]
        sample_count = int(samples.shape[0])

        ref_centered = ref_samples - ref_samples.mean(dim=0)
        candidate_centered = candidate_samples - candidate_samples.mean(dim=0)
        ref_flat = ref_centered.reshape(
            sample_count,
            *batch_shape,
            -1,
        )
        candidate_flat = candidate_centered.reshape(
            sample_count,
            *batch_shape,
            -1,
        )
        denominator = max(sample_count - 1, 1)
        cross_covariance = torch.einsum(
            "s...i,s...j->...ij",
            ref_flat,
            candidate_flat,
        ) / denominator
        candidate_covariance = torch.einsum(
            "s...i,s...j->...ij",
            candidate_flat,
            candidate_flat,
        ) / denominator
        candidate_noise = stats.expected_observation_variance[
            ..., n_ref:, :
        ].reshape(*batch_shape, -1)
        system = candidate_covariance + torch.diag_embed(
            candidate_noise.clamp_min(self.eps)
        )
        system = 0.5 * (system + system.transpose(-1, -2))
        eye = torch.eye(
            system.shape[-1],
            device=system.device,
            dtype=system.dtype,
        )
        system = system + max(self.jitter, self.eps) * eye
        solved = torch.linalg.solve(
            system,
            cross_covariance.transpose(-1, -2),
        )
        reduction_flat = (
            cross_covariance * solved.transpose(-1, -2)
        ).sum(dim=-1).clamp_min(0)
        num_outputs = samples.shape[-1]
        reduction = reduction_flat.reshape(
            *batch_shape,
            n_ref,
            num_outputs,
        )
        score = self._output_reduction(reduction)
        return self._finalize_joint_score(score, Xq)


class qNonGaussianNegIntegratedResponseMeanVariance(AcquisitionFunction):
    """Use fantasy NIPV when explicitly supported, otherwise use the proxy.

    Gamma base / mixed models currently expose a response-aware ``fantasize``
    implementation. Other non-Gaussian families fall back to the differentiable
    covariance-reduction proxy rather than silently using incompatible fantasy
    observations.
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
        fallback_to_proxy: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model)
        explicit_capability = getattr(
            model,
            "supports_non_gaussian_nipv",
            None,
        )
        if explicit_capability is None:
            explicit_capability = (
                type(model).__name__ in {"GammaGPModel", "GammaMixedGPModel"}
                and ".non_gaussian.gamma.base." in type(model).__module__
                and callable(getattr(model, "fantasize", None))
            )
        self._uses_proxy = not bool(explicit_capability)

        if not self._uses_proxy:
            self.acqf = qRegressionNegIntegratedPosteriorVariance(
                model=model,
                mc_points=mc_points,
                sampler=sampler,
                objective=objective,
                posterior_transform=posterior_transform,
                X_pending=X_pending,
                fallback_to_proxy=False,
                **kwargs,
            )
            return

        if not fallback_to_proxy:
            raise NotImplementedError(
                f"{type(model).__name__} does not declare support for "
                "non-Gaussian response-aware fantasization."
            )
        ignored = []
        if sampler is not None:
            ignored.append("sampler")
        if posterior_transform is not None:
            ignored.append("posterior_transform")
        suffix = ""
        if ignored:
            suffix = f" Ignored by proxy: {', '.join(ignored)}."
        warnings.warn(
            f"{type(model).__name__} does not support response-aware fantasy "
            "NIPV; using qNonGaussianIntegratedResponseMeanVarianceProxy."
            f"{suffix}",
            RuntimeWarning,
            stacklevel=2,
        )
        proxy_kwargs = dict(kwargs)
        if objective is not None:
            proxy_kwargs.setdefault("objective", objective)
        if X_pending is not None:
            proxy_kwargs.setdefault("X_pending", X_pending)
        self.acqf = qNonGaussianIntegratedResponseMeanVarianceProxy(
            model=model,
            mc_points=mc_points,
            **proxy_kwargs,
        )

    @property
    def uses_proxy(self) -> bool:
        """Return whether the covariance-reduction proxy is active."""
        return self._uses_proxy

    def set_X_pending(self, X_pending: Tensor | None = None) -> None:
        """Delegate pending-point updates."""
        if hasattr(self.acqf, "set_X_pending"):
            self.acqf.set_X_pending(X_pending)
        else:
            self.acqf.X_pending = X_pending

    def forward(self, X: Tensor) -> Tensor:
        """Evaluate fantasy NIPV or its response-mean proxy."""
        return self.acqf(X)


def _joint_bald_from_stats(
    stats,
    *,
    eps: float,
    jitter: float,
    num_flat_outputs: int | None = None,
) -> Tensor:
    """Evaluate joint variance-ratio information from response samples."""
    samples = stats.response_mean_samples
    sample_count = int(samples.shape[0])
    flat = samples.reshape(
        sample_count,
        *samples.shape[1:-2],
        -1,
    )
    if num_flat_outputs is not None:
        flat = flat[..., :num_flat_outputs]
    centered = flat - flat.mean(dim=0)
    covariance = torch.einsum(
        "s...i,s...j->...ij",
        centered,
        centered,
    ) / max(sample_count - 1, 1)
    noise = stats.expected_observation_variance.reshape(
        *stats.expected_observation_variance.shape[:-2],
        -1,
    )
    if num_flat_outputs is not None:
        noise = noise[..., :num_flat_outputs]
    noise = noise.clamp_min(eps)
    scaled = covariance / torch.sqrt(
        noise.unsqueeze(-1) * noise.unsqueeze(-2)
    )
    eye = torch.eye(
        scaled.shape[-1],
        device=scaled.device,
        dtype=scaled.dtype,
    )
    return 0.5 * safe_logdet(
        eye + scaled,
        jitter=max(jitter, eps),
    )


class qNonGaussianJointBALDProxy(_NonGaussianActiveLearningBase):
    """Joint moment BALD proxy preserving q/output response covariance."""

    def __init__(self, model, *, jitter: float = 1e-6, **kwargs: Any) -> None:
        if jitter <= 0:
            raise ValueError("jitter must be positive.")
        super().__init__(model, **kwargs)
        self.jitter = float(jitter)

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        """Evaluate batch information gain with joint response covariance."""
        score = _joint_bald_from_stats(
            self._stats(X),
            eps=self.eps,
            jitter=self.jitter,
        )
        return self._finalize_joint_score(score, X)


class qNonGaussianGreedyJointBALDProxy(qNonGaussianJointBALDProxy):
    """Marginal joint BALD gain of the final point conditioned on its prefix.

    For an ordered q-batch ``[x_1, ..., x_q]`` this returns
    ``I(x_1:q) - I(x_1:q-1)``. It therefore supports true greedy selection by
    placing already selected points before the candidate being optimized.
    """

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        """Evaluate the incremental information contributed by the last point."""
        Xq = ensure_q_batch(X)
        stats = self._stats(Xq)
        full = _joint_bald_from_stats(
            stats,
            eps=self.eps,
            jitter=self.jitter,
        )
        q = int(Xq.shape[-2])
        if q == 1:
            marginal = full
        else:
            num_outputs = int(stats.response_mean_samples.shape[-1])
            prefix = _joint_bald_from_stats(
                stats,
                eps=self.eps,
                jitter=self.jitter,
                num_flat_outputs=(q - 1) * num_outputs,
            )
            marginal = (full - prefix).clamp_min(0)
        return self._finalize_joint_score(marginal, Xq)


qNonGaussianPosteriorVariance = qNonGaussianResponseMeanVariance
qNonGaussianVariance = qNonGaussianResponseMeanVariance
qNonGaussianNegIntegratedPosteriorVariance = (
    qNonGaussianNegIntegratedResponseMeanVariance
)
qNonGaussianNIPV = qNonGaussianNegIntegratedResponseMeanVariance

__all__ = [name for name in globals() if name.startswith("qNonGaussian")]
