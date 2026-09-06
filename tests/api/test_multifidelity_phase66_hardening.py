from __future__ import annotations

import pytest
import torch

from bochan.api.acquisition.multifidelity_hvkg import _bundle_is_multi_output as hvkg_multi_output
from bochan.api.acquisition.multifidelity_momf import _bundle_is_multi_output as momf_multi_output
from bochan.api.configs import ModelBundle, ModelConfig
from bochan.models.multifidelity import (
    CostNormalizedTrace,
    best_objective_trace,
    cumulative_cost,
    hypervolume_regret_trace,
    hypervolume_trace,
    inference_hv_regret_cost_trace,
    multi_objective_cost_trace,
    single_objective_cost_trace,
)


class _TwoOutputModel:
    num_outputs = 2


class _OneOutputModel:
    num_outputs = 1


def _bundle(model, *, metadata=None) -> ModelBundle:
    return ModelBundle(
        model=model,
        train_X=torch.zeros(3, 2, dtype=torch.double),
        train_Y=torch.zeros(3, getattr(model, "num_outputs", 1), dtype=torch.double),
        model_config=ModelConfig(task_type="regression", model_type="multifidelity_gp"),
        model_type="multifidelity_gp",
        metadata={} if metadata is None else dict(metadata),
    )


def test_mfhvkg_and_momf_infer_correlated_multi_output_without_metadata_flag():
    bundle = _bundle(_TwoOutputModel())

    assert hvkg_multi_output(bundle) is True
    assert momf_multi_output(bundle) is True


def test_multi_output_metadata_remains_backward_compatible():
    bundle = _bundle(_OneOutputModel(), metadata={"multi_output": True})

    assert hvkg_multi_output(bundle) is True
    assert momf_multi_output(bundle) is True


def test_single_output_without_metadata_is_rejected_by_multi_output_detection():
    bundle = _bundle(_OneOutputModel())

    assert hvkg_multi_output(bundle) is False
    assert momf_multi_output(bundle) is False


def test_single_objective_cost_trace_aligns_incumbent_to_cumulative_cost():
    trace = single_objective_cost_trace(
        strategy="mfkg",
        values=torch.tensor([0.2, 0.1, 0.7, 0.6], dtype=torch.double),
        costs=torch.tensor([1.0, 2.0, 4.0, 1.0], dtype=torch.double),
    )

    assert isinstance(trace, CostNormalizedTrace)
    assert trace.metric_name == "best_objective"
    assert torch.equal(trace.cumulative_cost, torch.tensor([1.0, 3.0, 7.0, 8.0], dtype=torch.double))
    assert torch.equal(trace.metric, torch.tensor([0.2, 0.2, 0.7, 0.7], dtype=torch.double))


def test_minimize_incumbent_trace_uses_cumulative_minimum():
    values = torch.tensor([3.0, 4.0, 2.0, 2.5], dtype=torch.double)
    assert torch.equal(
        best_objective_trace(values, maximize=False),
        torch.tensor([3.0, 3.0, 2.0, 2.0], dtype=torch.double),
    )


def test_hypervolume_and_regret_traces_are_cost_normalized():
    Y = torch.tensor(
        [[1.0, 1.0], [2.0, 0.5], [1.5, 2.0]],
        dtype=torch.double,
    )
    costs = torch.tensor([1.0, 3.0, 2.0], dtype=torch.double)
    ref = torch.tensor([0.0, 0.0], dtype=torch.double)

    hv = hypervolume_trace(Y, ref_point=ref)
    trace = multi_objective_cost_trace(
        strategy="mfhvkg",
        Y=Y,
        costs=costs,
        ref_point=ref,
    )
    regret = inference_hv_regret_cost_trace(
        strategy="mfhvkg",
        hypervolume=hv,
        costs=costs,
        reference_hypervolume=float(hv[-1] + 1.0),
    )

    assert hv.shape == (3,)
    assert bool((hv[1:] >= hv[:-1]).all())
    assert torch.equal(trace.cumulative_cost, torch.tensor([1.0, 4.0, 6.0], dtype=torch.double))
    assert torch.equal(trace.metric, hv)
    assert regret.metric_name == "inference_hypervolume_regret"
    assert torch.allclose(regret.metric, hv[-1] + 1.0 - hv)


def test_benchmark_validation_rejects_invalid_costs_and_lengths():
    with pytest.raises(ValueError, match="strictly positive"):
        cumulative_cost([1.0, 0.0])
    with pytest.raises(ValueError, match="same length"):
        single_objective_cost_trace(strategy="mfmes", values=[1.0, 2.0], costs=[1.0])
    with pytest.raises(ValueError, match="non-negative"):
        hypervolume_regret_trace([1.0, 2.0], reference_hypervolume=-1.0)
