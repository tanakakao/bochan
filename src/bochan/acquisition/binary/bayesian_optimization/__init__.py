from __future__ import annotations

import torch
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.acquisition.multi_objective.objective import (
    IdentityMCMultiOutputObjective,
    MCMultiOutputObjective,
    WeightedMCMultiOutputObjective,
)
from botorch.sampling.normal import SobolQMCNormalSampler
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
)
from .hetero_multi_output import (
    qHeteroMultiOutputBinaryNParEGO as _BaseHeteroBinaryNParEGO,
)
from .hetero_multi_output_stable import (
    qHeteroMultiOutputBinaryExpectedHypervolumeImprovement,
)
from .hetero_single_output import (
    qHeteroBinaryExpectedImprovement,
    qHeteroBinaryProbabilityOfImprovement,
    qHeteroBinaryUpperConfidenceBound,
)

# Keep q=1 sequential optimization shape handling aligned across classification
# and ordinal NParEGO implementations.
_multi_output._reduce_sample_and_q_to_tbatch = (
    reduce_nparego_sample_and_q_to_tbatch
)

# Apply the same model-aware qNEHVI default used by ordinal models. This keeps
# Kronecker binary models out of BoTorch's insupported cached-Cholesky path.
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


class _OneToManyObjectiveAdapter(MCMultiOutputObjective):
    """Align objective ``X`` with one-to-many expanded posterior samples.

    Plain multi-output objectives require an expanded ``X`` so BoTorch's q-shape
    validation sees the same number of points as the posterior samples. An
    input-perturbation objective with ``n_w > 1`` is different: it intentionally
    needs the raw ``X`` so it can aggregate ``q * n_w`` samples back to ``q``.
    """

    def __init__(self, objective: MCMultiOutputObjective) -> None:
        super().__init__()
        self.objective = objective
        self._verify_output_shape = False

    def forward(self, samples: Tensor, X: Tensor | None = None) -> Tensor:
        n_w = getattr(self.objective, "n_w", None)
        aggregates_one_to_many = n_w is not None and int(n_w) > 1
        if aggregates_one_to_many:
            return self.objective(samples, X=X)

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


class _SequentialMCMultiOutputObjective(MCMultiOutputObjective):
    """Apply perturbation preprocessing before an output weighting objective."""

    def __init__(
        self,
        preprocessor: MCMultiOutputObjective,
        objective: MCMultiOutputObjective,
    ) -> None:
        super().__init__()
        self.preprocessor = _OneToManyObjectiveAdapter(preprocessor)
        self.objective = _OneToManyObjectiveAdapter(objective)

    def forward(self, samples: Tensor, X: Tensor | None = None) -> Tensor:
        values = self.preprocessor(samples, X=X)
        return self.objective(values, X=X)


class qHeteroMultiOutputBinaryNParEGO(_BaseHeteroBinaryNParEGO):
    """Heteroscedastic binary NParEGO with perturbation preprocessing."""

    def __init__(
        self,
        model,
        X_baseline: Tensor,
        ref_point: Tensor,
        *,
        weights: Tensor | None = None,
        sampler: SobolQMCNormalSampler | None = None,
        beta: float = 1.0,
        noise_penalty: float = 0.3,
        default_sigma: float = 0.0,
        noise_is_log_var: bool = True,
        samples_are_probs: bool = False,
        apply_sigmoid_if_needed: bool = True,
        eps: float = 1e-6,
        objective: MCMultiOutputObjective | None = None,
        best_f=None,
    ) -> None:
        del best_f
        sampler = sampler or SobolQMCNormalSampler(sample_shape=torch.Size([128]))
        tkwargs = {"dtype": X_baseline.dtype, "device": X_baseline.device}
        m = int(ref_point.numel())
        if weights is None:
            raw_weights = torch.rand(m, **tkwargs)
            weights = raw_weights / raw_weights.sum().clamp_min(1e-12)
        else:
            weights = weights.to(**tkwargs)
            weights = weights / weights.sum().clamp_min(1e-12)

        preprocessor = objective or IdentityMCMultiOutputObjective()
        weighted = WeightedMCMultiOutputObjective(weights=weights)
        composed_objective = _SequentialMCMultiOutputObjective(
            preprocessor=preprocessor,
            objective=weighted,
        )
        MCAcquisitionFunction.__init__(
            self,
            model=model,
            sampler=sampler,
            objective=composed_objective,
        )
        self.base_objective = composed_objective
        self.beta = float(beta)
        self.noise_penalty = float(noise_penalty)
        self.default_sigma = float(default_sigma)
        self.noise_is_log_var = bool(noise_is_log_var)
        self.samples_are_probs = bool(samples_are_probs)
        self.apply_sigmoid_if_needed = bool(apply_sigmoid_if_needed)
        self.eps = float(eps)

        with torch.no_grad():
            y = _hetero_multi_output.compute_hetero_multi_output_classification_train_y(
                model,
                X_baseline,
                noise_penalty=noise_penalty,
                default_sigma=default_sigma,
                noise_is_log_var=noise_is_log_var,
                apply_sigmoid_if_needed=apply_sigmoid_if_needed,
                eps=eps,
            )
            X_obj = X_baseline.unsqueeze(0)
            values = y.unsqueeze(0).unsqueeze(0)
            obj_train = self.base_objective(values, X=X_obj).squeeze()
            self.register_buffer("best_value", obj_train.max())

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


class qMultiOutputBinaryNParEGO(_multi_output.qMultiOutputBinaryNParEGO):
    """Binary NParEGO with high-level API support."""

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


from ._utils import (
    compute_binary_best_f,
    compute_hetero_binary_classification_best_f,
)
from .multi_output import (
    qMultiOutputBinaryExpectedHypervolumeImprovement,
    qMultiOutputBinaryNoisyExpectedHypervolumeImprovement,
    qMultiOutputBinaryProbabilityOfFeasibility,
)
from .single_output import (
    QBatchMode,
    qBinaryExpectedImprovement,
    qBinaryProbabilityOfFeasibility,
    qBinaryProbabilityOfImprovement,
    qBinaryUpperConfidenceBound,
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
