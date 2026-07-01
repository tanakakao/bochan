from __future__ import annotations

import torch
from botorch.acquisition.multi_objective.objective import (
    IdentityMCMultiOutputObjective,
    MCMultiOutputObjective,
)
from torch import Tensor

from bochan.acquisition._nehvi_cache_root import patch_nehvi_cache_root_init

from . import multi_output as _multi_output
from .hetero_multi_output import (
    qHeteroMultiOutputBinaryNoisyExpectedHypervolumeImprovement,
    qHeteroMultiOutputBinaryNParEGO,
)
from .hetero_multi_output_compat import (
    qHeteroMultiOutputBinaryExpectedHypervolumeImprovement,
)

from .hetero_single_output import (
    qHeteroBinaryUpperConfidenceBound,
    qHeteroBinaryExpectedImprovement,
    qHeteroBinaryProbabilityOfImprovement,
)

# Apply the same model-aware qNEHVI default used by ordinal models. This keeps
# Kronecker binary models out of BoTorch's incompatible cached-Cholesky path.
patch_nehvi_cache_root_init(
    _multi_output.qMultiOutputBinaryNoisyExpectedHypervolumeImprovement
)


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
    """Binary NParEGO with one-to-many objective input alignment."""

    def __init__(
        self,
        *args,
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
            *args,
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
