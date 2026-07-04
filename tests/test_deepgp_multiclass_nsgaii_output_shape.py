from __future__ import annotations

from types import SimpleNamespace

import torch

import bochan.api  # noqa: F401 - installs multiclass objective compatibility
import bochan.optim.nsgaii_adapter as nsgaii_adapter
from bochan.acquisition.multiclass.bayesian_optimization.multi_output import (
    MulticlassTargetProbabilityObjective,
)
from bochan.optim.nsgaii_output_compat import adapt_nsgaii_outputs


class _FixedAcquisition:
    def __init__(self, values: torch.Tensor) -> None:
        self.values = values
        self.model = SimpleNamespace(num_outputs=2)

    def __call__(self, X=None):
        del X
        return self.values


def _multiclass_objective() -> MulticlassTargetProbabilityObjective:
    return MulticlassTargetProbabilityObjective(num_outputs=2)


def test_deepgp_model_list_sample_axis_is_averaged_after_objective() -> None:
    X = torch.rand(250, 1, 5, dtype=torch.double)
    logits = torch.randn(10, 250, 2, 3, dtype=torch.double)
    probabilities = torch.softmax(logits, dim=-1)
    acquisition, objective = adapt_nsgaii_outputs(
        _FixedAcquisition(probabilities),
        _multiclass_objective(),
    )

    raw_values = acquisition(X=X)
    values = objective(raw_values)

    utilities = torch.arange(3, dtype=torch.double)
    expected = (probabilities * utilities).sum(dim=-1).mean(dim=0).unsqueeze(-2)
    assert values.shape == torch.Size([250, 1, 2])
    torch.testing.assert_close(values, expected)


def test_deepgp_sample_axis_is_averaged_when_q_axis_is_present() -> None:
    X = torch.rand(250, 1, 5, dtype=torch.double)
    logits = torch.randn(10, 250, 1, 2, 3, dtype=torch.double)
    probabilities = torch.softmax(logits, dim=-1)
    acquisition, objective = adapt_nsgaii_outputs(
        _FixedAcquisition(probabilities),
        _multiclass_objective(),
    )

    values = objective(acquisition(X=X))

    utilities = torch.arange(3, dtype=torch.double)
    expected = (probabilities * utilities).sum(dim=-1).mean(dim=0)
    assert values.shape == torch.Size([250, 1, 2])
    torch.testing.assert_close(values, expected)


def test_identity_objective_restores_missing_singleton_q_axis() -> None:
    X = torch.rand(250, 1, 5, dtype=torch.double)
    model_samples = torch.randn(10, 250, 2, dtype=torch.double)
    acquisition, objective = adapt_nsgaii_outputs(
        _FixedAcquisition(model_samples),
        None,
    )

    values = objective(acquisition(X=X))

    assert values.shape == torch.Size([250, 1, 2])
    torch.testing.assert_close(values, model_samples.mean(dim=0).unsqueeze(-2))


def test_public_nsgaii_adapter_passes_shape_compatible_objective(monkeypatch) -> None:
    X_eval = torch.rand(250, 1, 5, dtype=torch.double)
    logits = torch.randn(10, 250, 2, 3, dtype=torch.double)
    probabilities = torch.softmax(logits, dim=-1)
    target = _FixedAcquisition(probabilities)
    strategy = SimpleNamespace(
        model=target.model,
        objective=_multiclass_objective(),
        outcome_constraints=None,
        ref_point=None,
    )
    captured = {}

    monkeypatch.setattr(nsgaii_adapter, "_resolve_nsgaii_target", lambda _: target)

    def fake_optimize_acqf_nsgaii(**kwargs):
        raw_values = kwargs["acq_function"](X=X_eval)
        objective_values = kwargs["objective"](raw_values)
        captured["shape"] = objective_values.shape
        return (
            torch.zeros(3, 5, dtype=torch.double),
            torch.zeros(3, 2, dtype=torch.double),
        )

    monkeypatch.setattr(
        nsgaii_adapter._base,
        "optimize_acqf_nsgaii",
        fake_optimize_acqf_nsgaii,
    )

    nsgaii_adapter.optimize_acqf_nsgaii(
        acq_function=strategy,
        bounds=torch.tensor(
            [[0.0] * 5, [1.0] * 5],
            dtype=torch.double,
        ),
        q=3,
    )

    assert captured["shape"] == torch.Size([250, 1, 2])
