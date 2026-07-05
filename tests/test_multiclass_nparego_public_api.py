import torch

from bochan.api import (
    AcquisitionConfig,
    BayesianOptimizer,
    FitConfig,
    InputTransformConfig,
    ModelConfig,
)


def test_multiclass_multitask_nparego_uses_training_baseline() -> None:
    train_X = torch.tensor(
        [
            [0.0, 0.0],
            [0.2, 0.8],
            [0.4, 0.3],
            [0.6, 0.7],
            [0.8, 0.2],
            [1.0, 1.0],
        ],
        dtype=torch.double,
    )
    train_Y = torch.tensor(
        [
            [0.0, float("nan")],
            [0.0, 2.0],
            [1.0, 2.0],
            [1.0, 1.0],
            [2.0, float("nan")],
            [2.0, 0.0],
        ],
        dtype=torch.double,
    )
    optimizer = BayesianOptimizer(
        model_config=ModelConfig(
            task_type="multiclass",
            model_type="multitask",
            input_transform_config=InputTransformConfig(
                normalize=True,
                perturbation=False,
            ),
            outcome_transform=True,
            model_kwargs={"rank": 2},
        ),
        fit_config=FitConfig(skip_fit=True),
    )
    optimizer.fit(train_X, train_Y)
    optimizer.model.eval()
    optimizer.model.likelihood.eval()

    acquisition = optimizer.acquisition(
        AcquisitionConfig(name="nparego", acqf_kwargs={})
    )
    Xq = torch.tensor(
        [[[0.15, 0.25], [0.50, 0.50], [0.85, 0.75]]],
        dtype=torch.double,
        requires_grad=True,
    )
    value = acquisition(Xq)

    assert value.shape == torch.Size([1])
    assert torch.isfinite(value).all()
