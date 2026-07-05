from __future__ import annotations

from types import SimpleNamespace

import torch
from botorch.models.model import Model
from botorch.models.transforms.input import InputPerturbation
from torch import Tensor

from bochan.acquisition.multiclass.bayesian_optimization import (
    qMultiOutputMulticlassNoisyExpectedHypervolumeImprovement,
)
from bochan.acquisition.multiclass.bayesian_optimization.input_perturbation_compat import (
    InputPerturbationMultiOutputObjectiveAdapter,
)
from bochan.acquisition.multiclass.bayesian_optimization.multi_output import (
    MulticlassTargetProbabilityObjective,
)


class _BaselinePosteriorModel(Model):
    def __init__(self, *, n_w: int, num_classes: int = 3) -> None:
        super().__init__()
        self.input_transform = InputPerturbation(
            perturbation_set=torch.zeros(n_w, 2, dtype=torch.double)
        )
        self.num_classes = int(num_classes)

    @property
    def num_outputs(self) -> int:
        return 2

    def posterior(self, X: Tensor, *args, **kwargs):
        logits = torch.zeros(
            *X.shape[:-1],
            self.num_outputs,
            self.num_classes,
            device=X.device,
            dtype=X.dtype,
        )
        return SimpleNamespace(mean=torch.softmax(logits, dim=-1))


def test_qnehvi_baseline_passes_raw_x_to_input_perturbation_objective() -> None:
    n_baseline = 60
    n_w = 16
    model = _BaselinePosteriorModel(n_w=n_w)
    X_baseline = torch.rand(n_baseline, 2, dtype=torch.double)
    base_objective = MulticlassTargetProbabilityObjective(
        output_target_classes=[1, 2],
        num_outputs=2,
    )
    objective = InputPerturbationMultiOutputObjectiveAdapter(
        base_objective,
        n_w=n_w,
    )

    acquisition = qMultiOutputMulticlassNoisyExpectedHypervolumeImprovement(
        model=model,
        ref_point=torch.tensor([0.0, 0.0], dtype=torch.double),
        X_baseline=X_baseline,
        objective=objective,
    )

    assert acquisition.objective is objective
    assert acquisition.X_baseline.shape == torch.Size([n_baseline, 2])
