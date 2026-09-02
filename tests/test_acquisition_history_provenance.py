from types import SimpleNamespace

import torch

from bochan.api.acquisition.diagnostics import (
    build_acquisition_observation_diagnostics,
)
from bochan.api.acquisition.provenance import candidate_acquisition_diagnostics
from bochan.api.configs import (
    AcquisitionConfig,
    CandidateResult,
    DataContext,
    ModelBundle,
    ModelConfig,
    ObjectiveConfig,
    OptimizeConfig,
)


def _bundle(train_x, train_y):
    return ModelBundle(
        model=SimpleNamespace(output_names=["property"]),
        train_X=train_x,
        train_Y=train_y,
        model_config=ModelConfig(
            task_type="regression",
            model_type="base",
            outcome_transform=False,
        ),
        metadata={},
    )


def _candidate_result(context):
    return CandidateResult(
        candidates=torch.tensor([[0.25]], dtype=torch.double),
        acq_value=torch.tensor(1.0, dtype=torch.double),
        acqf=SimpleNamespace(),
        acq_config=AcquisitionConfig(name="qnei"),
        opt_config=OptimizeConfig(q=1),
        data_context=context,
    )


def test_candidate_context_retains_its_own_acquisition_diagnostics_snapshot():
    train_x = torch.tensor([[0.0], [1.0], [2.0]], dtype=torch.double)
    train_y = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.double)
    bundle = _bundle(train_x, train_y)
    config = AcquisitionConfig(
        name="qnei",
        objective_config=ObjectiveConfig(mode="scalar", output=0),
    )

    first_context = DataContext(X_baseline=train_x[:2], Y_baseline=train_y[:2])
    first_diagnostics = build_acquisition_observation_diagnostics(
        bundle=bundle,
        config=config,
        before_context=DataContext(),
        after_context=first_context,
    )
    first_result = _candidate_result(first_context)

    second_context = DataContext(X_baseline=train_x, Y_baseline=train_y)
    second_diagnostics = build_acquisition_observation_diagnostics(
        bundle=bundle,
        config=config,
        before_context=DataContext(),
        after_context=second_context,
    )
    second_result = _candidate_result(second_context)

    assert first_diagnostics["baseline_rows"] == 2
    assert second_diagnostics["baseline_rows"] == 3
    assert candidate_acquisition_diagnostics(first_result)["baseline_rows"] == 2
    assert candidate_acquisition_diagnostics(second_result)["baseline_rows"] == 3


def test_candidate_provenance_helper_returns_defensive_copy():
    train_x = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    train_y = torch.tensor([[1.0], [2.0]], dtype=torch.double)
    bundle = _bundle(train_x, train_y)
    context = DataContext(X_baseline=train_x, Y_baseline=train_y)
    build_acquisition_observation_diagnostics(
        bundle=bundle,
        config=AcquisitionConfig(name="qnei"),
        before_context=DataContext(),
        after_context=context,
    )
    result = _candidate_result(context)

    first = candidate_acquisition_diagnostics(result)
    assert first is not None
    first["baseline_rows"] = 999

    second = candidate_acquisition_diagnostics(result)
    assert second is not None
    assert second["baseline_rows"] == 2


def test_candidate_provenance_helper_is_backward_compatible_without_snapshot():
    result = _candidate_result(DataContext())

    assert candidate_acquisition_diagnostics(result) is None
