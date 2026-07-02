from __future__ import annotations

import torch
from botorch.acquisition.multi_objective.objective import (
    IdentityMCMultiOutputObjective,
    MCMultiOutputObjective,
)
from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor

from bochan.acquisition._nehvi_cache_root import patch_nehvi_cache_root_init
from bochan.acquisition._nparego_shape import (
    reduce_nparego_sample_and_q_to_tbatch,
)

from . import hetero_multi_output as _hetero_multi_output
from . import multi_output as _multi_output
from .hetero_multi_output import (
    qHeteroMultiOutputBinaryNoisyExpectedHypervolumeImprovement,
    qHeteroMultiOutputBinaryNParEGO as _BaseHeteroBinaryNParEGO,
)
from .hetero_multi_output_compat import (
    qHeteroMultiOutputBinaryExpectedHypervolumeImprovement,
)

from .hetero_single_output import (
    qHeteroBinaryUpperConfidenceBound,
    qHeteroBinaryExpectedImprovement,
    qHeteroBinaryProbabilityOfImprovement,
)

# Keep q=1 sequential optimization shape handling aligned across classification
# and ordinal NParEGO implementations.
_multi_output._reduce_sample_and_q_to_tbatch = (
    reduce_nparego_sample_and_q_to_tbatch
)

# Apply the same model-aware qNEHVI default used by ordinal models. This keeps
# Kronecker binary models out of BoTorch's incompatible cached-Cholesky path.
patch_nehvi_cache_root_init(
    _multi_output.qMultiOutputBinaryNoisyExpectedHypervolumeImprovement
)


def _normalized_hetero_objective_forward(
    self,
    samples: Tensor,
    X: Tensor | None = None,
) -> Tensor:
    """Normalize singleton q dimensions before applying the base objective."""
    if X is None:
        raise ValueError(
            "X must be provided for _HeteroClassificationMCMultiOutputObjective."
        )

    adjusted = _hetero_multi_output.hetero_adjust_classification_samples(
        self.model,
        X,
        samples,
        beta=self.beta,
        noise_penalty=self.noise_penalty,
        default_sigma=self.default_sigma,
        noise_is_log_var=self.noise_is_log_var,
        samples_are_probs=self.samples_are_probs,
        apply_sigmoid_if_needed=self.apply_sigmoid_if_needed,
        eps=self.eps,
    )

    x_q = int(X.shape[-2])
    if (
        adjusted.ndim >= 4
        and adjusted.shape[-2] == 1
        and adjusted.shape[-3] == x_q
    ):
        adjusted = adjusted.squeeze(-2)

    if self.base_objective is None:
        return adjusted
    return self.base_objective(adjusted, X=X)


_hetero_multi_output._HeteroClassificationMCMultiOutputObjective.forward = (
    _normalized_hetero_objective_forward
)


def _reduce_to_t_batch(values: Tensor, X: Tensor) -> Tensor:
    """Reduce MC/model/q dimensions while preserving ``X`` t-batch dimensions."""
    target_shape = tuple(int(size) for size in X.shape[:-2])
    q = int(X.shape[-2])

    q_dims = [i for i, size in enumerate(values.shape) if int(size) == q]
    if q_dims:
        values = values.max(dim=q_dims[-1]).values

    if not target_shape:
        return values.mean()

    shape = tuple(int(size) for size in values.shape)
    keep_dims: list[int] | None = None
    width = len(target_shape)
    for start in range(len(shape) - width + 1):
        if shape[start : start + width] == target_shape:
            keep_dims = list(range(start, start + width))
            break

    if keep_dims is None:
        raise RuntimeError(
            "Could not preserve the t-batch dimensions in hetero NParEGO: "
            f"values shape={shape}, expected t-batch shape={target_shape}."
        )

    reduce_dims = [dim for dim in range(values.ndim) if dim not in keep_dims]
    values = values.permute(*keep_dims, *reduce_dims)
    if reduce_dims:
        values = values.mean(dim=tuple(range(width, values.ndim)))
    return values


class qHeteroMultiOutputBinaryNParEGO(_BaseHeteroBinaryNParEGO):
    """Heteroscedastic binary NParEGO with scalar t-batch output."""

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        post = _hetero_multi_output.get_model_posterior(
            self.model,
            X,
            samples_are_probs=self.samples_are_probs,
        )
        samples = post.rsample(self.sampler.sample_shape)
        hetero = _hetero_multi_output.hetero_adjust_classification_samples(
            self.model,
            X,
            samples,
            beta=self.beta,
            noise_penalty=self.noise_penalty,
            default_sigma=self.default_sigma,
            noise_is_log_var=self.noise_is_log_var,
            samples_are_probs=self.samples_are_probs,
            apply_sigmoid_if_needed=self.apply_sigmoid_if_needed,
            eps=self.eps,
            posterior=post,
        )

        x_q = int(X.shape[-2])
        if (
            hetero.ndim >= 4
            and hetero.shape[-2] == 1
            and hetero.shape[-3] == x_q
        ):
            hetero = hetero.squeeze(-2)

        scalarized = self.base_objective(hetero, X=X)
        improvement = (scalarized - self.best_value.to(scalarized)).clamp_min(0.0)
        return _reduce_to_t_batch(improvement, X)


class _OneToManyObjectiveAdapter(MCMultiOutputObjective):
    """Align objective ``X`` with one-to-many expanded posterior samples.

    Binary models may expand each raw design point into ``n_w`` transformed
    points. BoTorch verifies that the objective output q-dimension matches the
    supplied ``X`` q-dimension, so the raw baseline ``X`` must be expanded before
    it is passed to an inner multi-output objective.
    """

    def __init__(self, objective: MCMultiOutputObjective) -> None:
        super().__init__()
        self.objective = objective
        self._verify_output_shape = False

    def forward(self, samples: Tensor, X: Tensor | None = None) -> Tensor:
        if X is not None:
            sample_q = int(samples.shape[-2])
            x_q = int(X.shape[-2])
            if sample_q != x_q:
                if x_q <= 0 or sample_q % x_q != 0:
                    raise RuntimeError(
                        "Cannot align one-to-many objective inputs: "
                        f"samples q={sample_q}, X q={x_q}."
                    )
                X = X.repeat_interleave(sample_q // x_q, dim=-2)
        return self.objective(samples, X=X)


class qMultiOutputBinaryNParEGO(_multi_output.qMultiOutputBinaryNParEGO):
    """Binary NParEGO with high-level API compatibility."""

    def __init__(
        self,
        model,
        X_baseline: Tensor,
        ref_point: Tensor,
        *,
        objective=None,
        best_f=None,
        **kwargs,
    ) -> None:
        # ``best_f`` may be injected by the high-level API's generic EI defaults,
        # but NParEGO computes and registers its own scalarized ``best_value`` from
        # ``X_baseline``. Accept and intentionally ignore the generic value.
        del best_f
        base_objective = (
            objective
            if objective is not None
            else IdentityMCMultiOutputObjective()
        )
        super().__init__(
            model=model,
            X_baseline=X_baseline,
            ref_point=ref_point,
            objective=_OneToManyObjectiveAdapter(base_objective),
            **kwargs,
        )


from .multi_output import (
    qMultiOutputBinaryProbabilityOfFeasibility,
    qMultiOutputBinaryExpectedHypervolumeImprovement,
    qMultiOutputBinaryNoisyExpectedHypervolumeImprovement,
)

from .single_output import (
    QBatchMode,
    qBinaryProbabilityOfFeasibility,
    qBinaryExpectedImprovement,
    qBinaryProbabilityOfImprovement,
    qBinaryUpperConfidenceBound,
)
from ._utils import (
    compute_binary_best_f,
    compute_hetero_binary_classification_best_f,
)

__all__ = [
    "QBatchMode",
    "qHeteroMultiOutputBinaryExpectedHypervolumeImprovement",
    "qHeteroMultiOutputBinaryNoisyExpectedHypervolumeImprovement",
    "qHeteroMultiOutputBinaryNParEGO",
    "qHeteroBinaryUpperConfidenceBound",
    "qHeteroBinaryExpectedImprovement",
    "qHeteroBinaryProbabilityOfImprovement",
    "qMultiOutputBinaryProbabilityOfFeasibility",
    "qMultiOutputBinaryExpectedHypervolumeImprovement",
    "qMultiOutputBinaryNoisyExpectedHypervolumeImprovement",
    "qMultiOutputBinaryNParEGO",
    "qBinaryProbabilityOfFeasibility",
    "qBinaryExpectedImprovement",
    "qBinaryProbabilityOfImprovement",
    "qBinaryUpperConfidenceBound",
    "compute_binary_best_f",
    "compute_hetero_binary_classification_best_f",
]
