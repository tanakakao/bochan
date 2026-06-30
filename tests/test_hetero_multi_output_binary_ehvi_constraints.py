from __future__ import annotations

import torch
from botorch.models import SingleTaskGP
from botorch.utils.multi_objective.box_decompositions import (
    FastNondominatedPartitioning,
)

from bochan.acquisition.binary.bayesian_optimization import (
    qHeteroMultiOutputBinaryExpectedHypervolumeImprovement,
)


def test_hetero_binary_ehvi_accepts_float_eta_with_constraints() -> None:
    train_X = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.double)
    train_Y = torch.tensor(
        [[0.1, 0.9], [0.5, 0.5], [0.9, 0.1]],
        dtype=torch.double,
    )
    model = SingleTaskGP(train_X, train_Y)
    ref_point = torch.tensor([0.0, 0.0], dtype=torch.double)
    partitioning = FastNondominatedPartitioning(
        ref_point=ref_point,
        Y=train_Y,
    )

    acqf = qHeteroMultiOutputBinaryExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point,
        partitioning=partitioning,
        constraints=[lambda samples: samples[..., 0] - 0.8],
        eta=1e-3,
        samples_are_probs=True,
    )

    assert torch.is_tensor(acqf.eta)
    assert acqf.eta.dtype == torch.double
    assert float(acqf.eta.reshape(-1)[0]) == 1e-3
