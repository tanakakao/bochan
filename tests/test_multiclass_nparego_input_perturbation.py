from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from botorch.models.model import Model
from botorch.models.transforms.input import InputPerturbation
from torch import Tensor

from bochan.acquisition.multiclass.bayesian_optimization import (
    qMultiOutputMulticlassNParEGO,
)
from bochan.acquisition.multiclass.bayesian_optimization.input_perturbation import (
    InputPerturbationMultiOutputObjectiveAdapter,
)
from bochan.acquisition.multiclass.bayesian_optimization.multi_output import (
    MulticlassTargetProbabilityObjective,
)


class _NParEGOPosteriorModel(Model):
    def __init__(self, *, n_w: int, expand_posterior: bool, num_classes: int = 3) -> None:
        super().__init__()
        self.input_transform = InputPerturbation(perturbation_set=torch.zeros(n_w, 2, dtype=torch.double))
        self.n_w = int(n_w)
        self.expand_posterior = bool(expand_posterior)
        self.num_classes = int(num_classes)

    @property
    def num_outputs(self) -> int:
        return 2

    def posterior(self, X: Tensor, *args, **kwargs) -> SimpleNamespace:
        q = int(X.shape[-2]) * (self.n_w if self.expand_posterior else 1)
        logits = torch.zeros(
            *X.shape[:-2],
            q,
            self.num_outputs,
            self.num_classes,
            device=X.device,
            dtype=X.dtype,
        )
        return SimpleNamespace(mean=torch.softmax(logits, dim=-1))


def _objective(n_w: int) -> InputPerturbationMultiOutputObjectiveAdapter:
    return InputPerturbationMultiOutputObjectiveAdapter(
        MulticlassTargetProbabilityObjective(
            output_target_classes=[1, 2],
            num_outputs=2,
        ),
        n_w=n_w,
    )


@pytest.mark.parametrize("expand_posterior", [False, True])
def test_nparego_passes_raw_x_to_input_perturbation_objective(
    expand_posterior: bool,
) -> None:
    n_baseline = 60
    n_w = 16
    q = 3
    batch_size = 32
    sample_shape = 128
    model = _NParEGOPosteriorModel(n_w=n_w, expand_posterior=expand_posterior)
    objective = _objective(n_w)
    acquisition = qMultiOutputMulticlassNParEGO(
        model=model,
        X_baseline=torch.rand(n_baseline, 2, dtype=torch.double),
        ref_point=torch.tensor([0.0, 0.0], dtype=torch.double),
        objective=objective,
    )

    q_like = q * n_w if expand_posterior else q
    logits = torch.zeros(
        sample_shape,
        batch_size,
        q_like,
        2,
        3,
        dtype=torch.double,
    )
    samples = torch.softmax(logits, dim=-1)
    X = torch.rand(batch_size, q, 2, dtype=torch.double)

    with patch.object(acquisition, "get_posterior_samples", return_value=samples):
        values = acquisition(X)

    assert acquisition.base_objective is objective
    assert acquisition.objective is objective
    assert values.shape == torch.Size([batch_size])
