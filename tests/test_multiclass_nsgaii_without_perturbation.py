from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

import bochan.api
import bochan.api.engine as engine_module
from bochan.acquisition.multiclass.bayesian_optimization.multi_output import (
    MulticlassTargetProbabilityObjective,
)
from bochan.api import AcquisitionConfig, InputTransformConfig, ModelConfig
from bochan.optim.nsgaii_strategy import build_nsgaii_strategy


class _DummyMulticlassModel(nn.Module):
    @property
    def num_outputs(self) -> int:
        return 2


def _make_bundle():
    model_config = ModelConfig(
        task_type="multiclass",
        model_type="base",
        input_transform_config=InputTransformConfig(
            normalize=True,
            perturbation=False,
        ),
        outcome_transform=True,
    )
    return SimpleNamespace(
        task_type="multiclass",
        model_type="base",
        model=_DummyMulticlassModel(),
        model_config=model_config,
        train_X=torch.rand(8, 5, dtype=torch.double),
        train_Y=torch.zeros(8, 2, dtype=torch.long),
        metadata={"multi_output": True},
    )


def test_multiclass_nsgaii_infers_objective_without_input_perturbation() -> None:
    bundle = _make_bundle()
    resolved = engine_module._resolve_objective_config_n_w_from_input_transform(
        acq_config=AcquisitionConfig(name="nsgaii"),
        bundle=bundle,
    )

    assert resolved.objective_config is not None
    assert resolved.objective_config.mode == "multi_output"
    assert resolved.objective_config.n_w is None


def test_multiclass_nsgaii_strategy_reduces_class_probability_axis() -> None:
    bundle = _make_bundle()
    resolved = engine_module._resolve_objective_config_n_w_from_input_transform(
        acq_config=AcquisitionConfig(name="nsgaii"),
        bundle=bundle,
    )
    strategy = build_nsgaii_strategy(
        bundle=bundle,
        config=resolved,
        data_context=SimpleNamespace(constraints=None, ref_point=None),
    )

    assert isinstance(strategy.objective, MulticlassTargetProbabilityObjective)

    logits = torch.randn(250, 1, 2, 3, dtype=torch.double)
    probabilities = torch.softmax(logits, dim=-1)
    X = torch.rand(250, 1, 5, dtype=torch.double)

    values = strategy.objective(probabilities, X=X)
    utilities = torch.arange(3, dtype=torch.double)
    expected = (probabilities * utilities).sum(dim=-1)

    assert values.shape == torch.Size([250, 1, 2])
    torch.testing.assert_close(values, expected)


def test_multiclass_score_strategy_stays_without_vector_objective() -> None:
    bundle = _make_bundle()
    resolved = engine_module._resolve_objective_config_n_w_from_input_transform(
        acq_config=AcquisitionConfig(name="entropy"),
        bundle=bundle,
    )

    assert resolved.objective_config is None
