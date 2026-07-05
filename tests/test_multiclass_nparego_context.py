from __future__ import annotations

import inspect

import torch

from bochan.acquisition.binary.bayesian_optimization import qMultiOutputBinaryNParEGO
from bochan.acquisition.multiclass.bayesian_optimization import qMultiOutputMulticlassNParEGO
from bochan.acquisition.ordinal.bayesian_optimization import qMultiOutputOrdinalNParEGO
from bochan.api import (
    AcquisitionConfig,
    BayesianOptimizer,
    FitConfig,
    InputTransformConfig,
    ModelConfig,
)
from bochan.optim.nsgaii_strategy import NSGAIIStrategy


def test_classification_nparego_signatures_preserve_context_and_constraints() -> None:
    for acquisition_cls in (
        qMultiOutputBinaryNParEGO,
        qMultiOutputOrdinalNParEGO,
        qMultiOutputMulticlassNParEGO,
    ):
        parameters = inspect.signature(acquisition_cls).parameters
        assert "model" in parameters
        assert "X_baseline" in parameters
        assert "ref_point" in parameters
        assert "constraints" in parameters
        assert "eta" in parameters
        assert "fat" in parameters


def test_public_multiclass_multitask_acquisitions_accept_missing_targets() -> None:
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

    Xq = torch.tensor(
        [[[0.15, 0.25], [0.50, 0.50], [0.85, 0.75]]],
        dtype=torch.double,
        requires_grad=True,
    )

    for name in (
        "ehvi",
        "nehvi",
        "nparego",
        "bald",
        "entropy",
        "variance",
        "straddle",
        "icu",
        "nsgaii",
    ):
        acquisition = optimizer.acquisition(
            AcquisitionConfig(name=name, acqf_kwargs={})
        )
        if name == "nsgaii":
            assert isinstance(acquisition, NSGAIIStrategy)
            continue

        value = acquisition(Xq)
        assert torch.isfinite(value).all(), name
