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

    acquisition = optimizer.acquisition(
        AcquisitionConfig(name="nparego", acqf_kwargs={})
    )

    assert acquisition.model is optimizer.model
    assert acquisition.ref_point.shape == torch.Size([2])
    assert torch.isfinite(acquisition.ref_point).all()
    assert torch.isfinite(acquisition.best_value).all()
