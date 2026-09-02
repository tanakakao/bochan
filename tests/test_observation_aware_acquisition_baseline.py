from types import SimpleNamespace

import pytest
import torch

from bochan.api.acquisition.defaults.observations import (
    resolve_observation_aware_baselines,
)
from bochan.api.configs import (
    AcquisitionConfig,
    DataContext,
    ModelBundle,
    ModelConfig,
    ObjectiveConfig,
)


def _bundle(train_y, *, output_names=None):
    train_x = torch.arange(8, dtype=torch.double).reshape(4, 2)
    model = SimpleNamespace(output_names=output_names)
    return ModelBundle(
        model=model,
        train_X=train_x,
        train_Y=torch.as_tensor(train_y, dtype=torch.double),
        model_config=ModelConfig(
            task_type="regression",
            model_type="base",
            outcome_transform=False,
        ),
        metadata={},
    )


def test_scalar_output_baseline_uses_only_rows_observed_for_that_output():
    bundle = _bundle(
        [
            [1.0, 10.0],
            [float("nan"), 11.0],
            [3.0, float("nan")],
            [float("nan"), 13.0],
        ]
    )
    context = DataContext(
        X_baseline=bundle.train_X,
        Y_baseline=bundle.train_Y,
    )
    config = AcquisitionConfig(
        name="qnei",
        objective_config=ObjectiveConfig(mode="scalar", output=0),
    )

    resolved = resolve_observation_aware_baselines(bundle, config, context)

    assert torch.equal(resolved.X_baseline, bundle.train_X[[0, 2]])
    assert torch.equal(resolved.Y_baseline[:, 0], torch.tensor([1.0, 3.0], dtype=torch.double))
    assert torch.isnan(resolved.Y_baseline[1, 1])


def test_named_scalar_output_uses_model_output_names():
    bundle = _bundle(
        [
            [1.0, 10.0],
            [float("nan"), 11.0],
            [3.0, float("nan")],
            [4.0, 13.0],
        ],
        output_names=["strength", "conductivity"],
    )
    context = DataContext(
        X_baseline=bundle.train_X,
        Y_baseline=bundle.train_Y,
    )
    config = AcquisitionConfig(
        name="qnei",
        objective_config=ObjectiveConfig(mode="scalar", output="conductivity"),
    )

    resolved = resolve_observation_aware_baselines(bundle, config, context)

    assert torch.equal(resolved.X_baseline, bundle.train_X[[0, 1, 3]])
    assert torch.isfinite(resolved.Y_baseline[:, 1]).all()


def test_scalarized_outputs_require_jointly_observed_rows():
    bundle = _bundle(
        [
            [1.0, 10.0],
            [2.0, float("nan")],
            [float("nan"), 12.0],
            [4.0, 13.0],
        ]
    )
    context = DataContext(
        X_baseline=bundle.train_X,
        Y_baseline=bundle.train_Y,
    )
    config = AcquisitionConfig(
        name="qnei",
        objective_config=ObjectiveConfig(
            mode="scalar",
            outputs=[0, 1],
            weights=[0.5, 0.5],
        ),
    )

    resolved = resolve_observation_aware_baselines(bundle, config, context)

    assert torch.equal(resolved.X_baseline, bundle.train_X[[0, 3]])
    assert torch.isfinite(resolved.Y_baseline[:, [0, 1]]).all()


def test_multi_output_acquisition_keeps_partial_baseline_for_posterior_use():
    bundle = _bundle(
        [
            [1.0, 10.0],
            [float("nan"), 11.0],
            [3.0, float("nan")],
            [4.0, 13.0],
        ]
    )
    context = DataContext(
        X_baseline=bundle.train_X,
        Y_baseline=bundle.train_Y,
    )
    config = AcquisitionConfig(
        name="qnehvi",
        objective_config=ObjectiveConfig(mode="multi_output", outputs=[0, 1]),
    )

    resolved = resolve_observation_aware_baselines(bundle, config, context)

    assert resolved is context
    assert resolved.X_baseline is bundle.train_X
    assert resolved.Y_baseline is bundle.train_Y


def test_explicit_custom_baseline_is_not_overwritten():
    bundle = _bundle(
        [
            [1.0, 10.0],
            [float("nan"), 11.0],
            [3.0, float("nan")],
            [4.0, 13.0],
        ]
    )
    custom_x = bundle.train_X[[1, 3]].clone()
    custom_y = bundle.train_Y[[1, 3]].clone()
    context = DataContext(X_baseline=custom_x, Y_baseline=custom_y)
    config = AcquisitionConfig(
        name="qnei",
        objective_config=ObjectiveConfig(mode="scalar", output=0),
    )

    resolved = resolve_observation_aware_baselines(bundle, config, context)

    assert resolved is context
    assert resolved.X_baseline is custom_x
    assert resolved.Y_baseline is custom_y


def test_complete_legacy_data_keeps_existing_baseline_identity():
    bundle = _bundle(
        [
            [1.0, 10.0],
            [2.0, 11.0],
            [3.0, 12.0],
            [4.0, 13.0],
        ]
    )
    context = DataContext(
        X_baseline=bundle.train_X,
        Y_baseline=bundle.train_Y,
    )
    config = AcquisitionConfig(
        name="qnei",
        objective_config=ObjectiveConfig(mode="scalar", output=0),
    )

    resolved = resolve_observation_aware_baselines(bundle, config, context)

    assert resolved is context
    assert resolved.X_baseline is bundle.train_X
    assert resolved.Y_baseline is bundle.train_Y


def test_tensor_list_complete_data_is_a_strict_noop():
    train_x = torch.arange(8, dtype=torch.double).reshape(4, 2)
    train_y = [
        torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.double),
        torch.tensor([[10.0], [11.0], [12.0], [13.0]], dtype=torch.double),
    ]
    bundle = ModelBundle(
        model=SimpleNamespace(output_names=["strength", "conductivity"]),
        train_X=train_x,
        train_Y=train_y,
        model_config=ModelConfig(
            task_type="regression",
            model_type="base",
            outcome_transform=False,
        ),
        metadata={},
    )
    context = DataContext(X_baseline=train_x, Y_baseline=train_y)
    config = AcquisitionConfig(
        name="qnei",
        objective_config=ObjectiveConfig(mode="scalar", output="strength"),
    )

    resolved = resolve_observation_aware_baselines(bundle, config, context)

    assert resolved is context
    assert resolved.X_baseline is train_x
    assert resolved.Y_baseline is train_y


def test_scalar_output_without_any_observation_raises_clear_error():
    bundle = _bundle(
        [
            [1.0, float("nan")],
            [2.0, float("nan")],
            [3.0, float("nan")],
            [4.0, float("nan")],
        ]
    )
    context = DataContext(
        X_baseline=bundle.train_X,
        Y_baseline=bundle.train_Y,
    )
    config = AcquisitionConfig(
        name="qnei",
        objective_config=ObjectiveConfig(mode="scalar", output=1),
    )

    with pytest.raises(ValueError, match="no jointly observed rows"):
        resolve_observation_aware_baselines(bundle, config, context)
