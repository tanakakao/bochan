from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from botorch.acquisition.acquisition import AcquisitionFunction

from bochan.acquisition.feasible import (
    FeasibilityWeightedAcquisition,
    OrdinalRankConstraintSpec,
)
from bochan.acquisition.regression.active_learning import (
    qMultiOutputRegressionBALD,
    qMultiOutputRegressionPosteriorVariance,
    qMultiOutputRegressionPredictiveEntropy,
)
from bochan.acquisition.regression.levelset_estimation import (
    qMultiOutputRegressionICU,
    qMultiOutputRegressionStraddle,
)
from bochan.api import AcquisitionConfig, ModelBundle, ModelConfig, OutcomeConstraintConfig
from bochan.api import engine as api_engine
from bochan.tabular import TabularBayesianOptimizer  # noqa: F401


class _PosteriorModel:
    num_outputs = 2

    def eval(self) -> None:
        return None

    def posterior(self, X, observation_noise=False):
        del observation_noise
        mean = torch.stack(
            [
                torch.full(X.shape[:-1], 0.7, dtype=X.dtype, device=X.device),
                torch.full(X.shape[:-1], 1.0, dtype=X.dtype, device=X.device),
            ],
            dim=-1,
        )
        return SimpleNamespace(mean=mean, variance=torch.ones_like(mean))


class _OrdinalProbabilityModel(_PosteriorModel):
    output_names = ["property", "quality"]

    def class_probs_list(self, X, output_indices=None):
        del output_indices
        probs = torch.zeros(*X.shape[:-1], 3, dtype=X.dtype, device=X.device)
        probs[..., 0] = 0.1
        probs[..., 1] = 0.2
        probs[..., 2] = 0.7
        return [probs]


class _NativeConstraintAcquisition(AcquisitionFunction):
    def __init__(self, model, constraints=None) -> None:
        super().__init__(model=model)
        self.constraints = constraints

    def forward(self, X):
        return X.sum(dim=(-1, -2))


def _make_bundle(model=None) -> ModelBundle:
    return ModelBundle(
        model=_PosteriorModel() if model is None else model,
        train_X=torch.zeros(3, 2, dtype=torch.double),
        train_Y=torch.zeros(3, 2, dtype=torch.double),
        model_config=ModelConfig(
            task_type="regression",
            model_type="base",
            outcome_transform=False,
        ),
        task_type="regression",
        model_type="base",
        metadata={"multi_output": True},
    )


def _constraint_config() -> OutcomeConstraintConfig:
    return OutcomeConstraintConfig(
        output_indices=[0, 1],
        operators=["ge", "le"],
        thresholds=[0.5, 1.2],
    )


@pytest.mark.parametrize(
    "acqf_cls",
    [
        qMultiOutputRegressionBALD,
        qMultiOutputRegressionPredictiveEntropy,
        qMultiOutputRegressionPosteriorVariance,
        qMultiOutputRegressionStraddle,
        qMultiOutputRegressionICU,
    ],
)
def test_non_native_regression_acquisitions_use_feasibility_wrapper(acqf_cls) -> None:
    acqf = api_engine.build_acquisition(
        bundle=_make_bundle(),
        config=AcquisitionConfig(
            name=acqf_cls.__name__,
            acqf_cls=acqf_cls,
            outcome_constraint_config=_constraint_config(),
        ),
    )

    assert isinstance(acqf, FeasibilityWeightedAcquisition)
    assert isinstance(acqf.acqf, acqf_cls)
    assert len(acqf.constraints) == 2


def test_feasibility_wrapper_falls_back_to_standard_posterior_signature() -> None:
    acqf = api_engine.build_acquisition(
        bundle=_make_bundle(),
        config=AcquisitionConfig(
            name="bald",
            acqf_cls=qMultiOutputRegressionBALD,
            outcome_constraint_config=_constraint_config(),
        ),
    )

    values = acqf.constraint_values(torch.zeros(1, 2, 2, dtype=torch.double))
    expected = torch.tensor(
        [[[0.5 - 0.7, 1.0 - 1.2], [0.5 - 0.7, 1.0 - 1.2]]],
        dtype=torch.double,
    )
    torch.testing.assert_close(values, expected)


def test_native_constraint_acquisition_keeps_sample_constraints() -> None:
    config = AcquisitionConfig(
        name="native",
        acqf_cls=_NativeConstraintAcquisition,
        outcome_constraint_config=_constraint_config(),
    )

    acqf = api_engine.build_acquisition(bundle=_make_bundle(), config=config)

    assert isinstance(acqf, _NativeConstraintAcquisition)
    assert acqf.constraints is config.constraints


def test_model_dependent_ordinal_constraint_wraps_native_acquisition() -> None:
    config = AcquisitionConfig(
        name="native",
        acqf_cls=_NativeConstraintAcquisition,
        outcome_constraint_config=OutcomeConstraintConfig(
            constraints=[
                OrdinalRankConstraintSpec(
                    output="quality",
                    rank=1,
                    sense="ge",
                    probability_threshold=0.8,
                )
            ],
            eta=0.05,
        ),
    )

    assert config.constraints is None

    acqf = api_engine.build_acquisition(
        bundle=_make_bundle(model=_OrdinalProbabilityModel()),
        config=config,
    )

    assert isinstance(acqf, FeasibilityWeightedAcquisition)
    assert isinstance(acqf.acqf, _NativeConstraintAcquisition)
    values = acqf.constraint_values(torch.zeros(1, 2, 2, dtype=torch.double))
    expected = torch.full((1, 2, 1), -0.1, dtype=torch.double)
    torch.testing.assert_close(values, expected)
