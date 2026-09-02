from types import SimpleNamespace

import torch

from bochan.api import ObservationData
from bochan.api.acquisition.diagnostics import (
    build_acquisition_observation_diagnostics,
)
from bochan.api.configs import (
    AcquisitionConfig,
    DataContext,
    ModelBundle,
    ModelConfig,
    ObjectiveConfig,
)


def _bundle(train_x, train_y, *, output_names=None, train_yvar=None):
    return ModelBundle(
        model=SimpleNamespace(output_names=output_names),
        train_X=train_x,
        train_Y=train_y,
        train_Yvar=train_yvar,
        model_config=ModelConfig(
            task_type="regression",
            model_type="base",
            outcome_transform=False,
        ),
        metadata={},
    )


def test_diagnostics_report_filtered_partial_scalar_baseline_and_status_counts():
    observations = ObservationData.from_status(
        X=torch.arange(10, dtype=torch.double).reshape(5, 2),
        Y=torch.tensor(
            [
                [1.0, 10.0],
                [2.0, float("nan")],
                [float("nan"), 12.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ],
            dtype=torch.double,
        ),
        status=["success", "success", "success", "failed", "pending"],
    )
    train_x, train_y = observations.objective_training_data()
    bundle = _bundle(
        train_x,
        train_y,
        output_names=["strength", "conductivity"],
    )
    before = DataContext(X_baseline=train_x, Y_baseline=train_y)
    after = DataContext(
        X_baseline=train_x[[0, 1]],
        Y_baseline=train_y[[0, 1]],
    )
    config = AcquisitionConfig(
        name="qnei",
        objective_config=ObjectiveConfig(mode="scalar", output="strength"),
    )

    diagnostics = build_acquisition_observation_diagnostics(
        bundle=bundle,
        config=config,
        before_context=before,
        after_context=after,
        observations=observations,
    )

    assert diagnostics["training_rows"] == 3
    assert diagnostics["baseline_rows"] == 2
    assert diagnostics["baseline_source"] == "automatic"
    assert diagnostics["baseline_filtered"] is True
    assert diagnostics["partial_observation"] is True
    assert diagnostics["observed_per_output"] == [2, 2]
    assert diagnostics["objective_output_indices"] == [0]
    assert diagnostics["observation_rows"] == 5
    assert diagnostics["success_rows"] == 3
    assert diagnostics["failed_rows"] == 1
    assert diagnostics["pending_rows"] == 1
    assert diagnostics["failed_excluded_from_objective_training"] is True
    assert diagnostics["pending_excluded_from_objective_training"] is True


def test_diagnostics_preserve_explicit_complete_baseline_semantics():
    train_x = torch.arange(8, dtype=torch.double).reshape(4, 2)
    train_y = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.double)
    bundle = _bundle(train_x, train_y)
    explicit_x = train_x[[0, 2]].clone()
    explicit_y = train_y[[0, 2]].clone()
    context = DataContext(X_baseline=explicit_x, Y_baseline=explicit_y)
    config = AcquisitionConfig(
        name="qnei",
        objective_config=ObjectiveConfig(mode="scalar", output=0),
    )

    diagnostics = build_acquisition_observation_diagnostics(
        bundle=bundle,
        config=config,
        before_context=context,
        after_context=context,
    )

    assert diagnostics["baseline_source"] == "explicit"
    assert diagnostics["baseline_filtered"] is False
    assert diagnostics["partial_observation"] is False
    assert diagnostics["observed_per_output"] == [4]
    assert diagnostics["failure_rows"] if False else True
    assert diagnostics["failed_rows"] == 0
    assert diagnostics["pending_rows"] == 0
    assert diagnostics["failed_excluded_from_objective_training"] is False
    assert diagnostics["pending_excluded_from_objective_training"] is False
