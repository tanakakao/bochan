from __future__ import annotations

from types import MethodType

import torch

from bochan.acquisition.objective import OrdinalExpectedUtilityMCObjective
from bochan.acquisition.ordinal.bayesian_optimization import qOrdinalExpectedImprovement
from bochan.models.ordinal.base import OrdinalGPModel


def test_ordinal_ei_pointwise_uses_all_q_points() -> None:
    x_train = torch.rand(12, 2, dtype=torch.double)
    y_train = torch.tensor([0, 1, 2] * 4)
    model = OrdinalGPModel(x_train, y_train, num_classes=3)
    objective = OrdinalExpectedUtilityMCObjective(
        model.ordinal_likelihood,
        torch.tensor([0.0, 1.0, 2.0], dtype=torch.double),
    )
    x = torch.rand(1, 3, 2, dtype=torch.double)
    all_high = torch.full((4, 1, 3), 1.8, dtype=torch.double)
    one_high = torch.tensor([[[1.8, 0.2, 0.1]]], dtype=torch.double).expand(4, -1, -1)

    acqf = qOrdinalExpectedImprovement(
        model=model,
        objective=objective,
        best_f=1.0,
    )

    def bind(samples):
        def sample_method(self, X, *, name):
            return samples.to(X)
        acqf._posterior_samples_as_utility = MethodType(sample_method, acqf)

    bind(all_high)
    value_all = acqf(x)
    bind(one_high)
    value_one = acqf(x)

    assert value_all.item() > value_one.item()
