from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from bochan.acquisition.objective import (
    MultiOutputBinaryClassificationInputPerturbationObjective,
)
from bochan.api.automatic_multiobjective import observed_multiobjective_values
from bochan.api.configs import AcquisitionConfig, DataContext
from bochan.models.classification.binary.base import (
    MultiOutputBinaryClassificationModel,
)
from bochan.models.classification.binary.high_dim import (
    PCABinaryClassificationGPModel,
    REMBOBinaryClassificationGPModel,
)
from bochan.models.transforms.input import build_input_transform


@pytest.mark.parametrize(
    "model_cls",
    [PCABinaryClassificationGPModel, REMBOBinaryClassificationGPModel],
)
def test_projected_binary_automatic_observed_values_preserve_output_axis(
    model_cls,
) -> None:
    """Input perturbations must not be folded into EHVI's objective axis."""

    torch.manual_seed(7)
    n_train = 16
    n_w = 4
    n_outputs = 2
    train_X = torch.rand(n_train, 5, dtype=torch.double)
    train_Y = torch.randint(
        0,
        2,
        (n_train, n_outputs),
        dtype=torch.double,
    )
    bounds = torch.stack(
        (train_X.min(dim=0).values, train_X.max(dim=0).values)
    )

    submodels = [
        model_cls(
            train_X=train_X,
            train_Y=train_Y[:, output_index],
            input_transform=build_input_transform(
                train_X=train_X,
                bounds=bounds,
                perturbation=True,
                n_w=n_w,
                normalize=True,
            ),
            n_components=2,
        )
        for output_index in range(n_outputs)
    ]
    model = MultiOutputBinaryClassificationModel(*submodels)
    model.eval()

    objective = MultiOutputBinaryClassificationInputPerturbationObjective(
        n_w=n_w,
        risk_type=None,
    )
    bundle = SimpleNamespace(
        task_type="binary",
        model=model,
        train_X=train_X,
        train_Y=train_Y,
    )
    config = AcquisitionConfig(name="ehvi", objective=objective)

    values = observed_multiobjective_values(
        bundle,
        config,
        DataContext(),
    )

    posterior_mean = model.posterior(train_X.unsqueeze(0)).mean
    expected = posterior_mean.reshape(
        1,
        n_train,
        n_w,
        n_outputs,
    ).mean(dim=-2).squeeze(0)

    assert values.shape == torch.Size([n_train, n_outputs])
    assert torch.allclose(values, expected)
