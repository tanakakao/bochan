from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from bochan.api import AcquisitionConfig, DataContext, ModelBundle, ModelConfig
from bochan.api.acquisition import defaults as engine_defaults


class _NParEGOExpectedImprovement:
    def __init__(self, model, best_f, objective=None) -> None:
        self.model = model
        self.best_f = best_f
        self.objective = objective


def _make_bundle() -> ModelBundle:
    train_X = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.double)
    train_Y = torch.tensor(
        [[1.0, 3.0], [2.0, 2.0], [0.0, 4.0]],
        dtype=torch.double,
    )
    config = ModelConfig(
        task_type="regression",
        model_type="base",
        outcome_transform=False,
    )
    return ModelBundle(
        model=SimpleNamespace(num_outputs=2),
        train_X=train_X,
        train_Y=train_Y,
        model_config=config,
        task_type="regression",
        model_type="base",
    )


def test_nparego_creates_scalarization_before_computing_best_f(monkeypatch) -> None:
    bundle = _make_bundle()
    expected = torch.tensor(1.75, dtype=torch.double)
    calls = []

    def fake_compute_best_f(bundle_arg, config_arg, context_arg):
        calls.append((bundle_arg, config_arg, context_arg))
        assert config_arg.objective is not None
        return expected

    monkeypatch.setattr(engine_defaults, "compute_best_f", fake_compute_best_f)

    resolved, context = engine_defaults.resolve_acquisition_defaults(
        bundle,
        AcquisitionConfig(
            name="nparego",
            acqf_cls=_NParEGOExpectedImprovement,
        ),
        DataContext(),
    )

    assert len(calls) == 1
    assert resolved.objective is not None
    assert context.best_f is expected
    assert "best_f" not in resolved.acqf_kwargs


def test_default_nparego_objective_scalarizes_multi_output_values() -> None:
    bundle = _make_bundle()

    resolved, context = engine_defaults.resolve_acquisition_defaults(
        bundle,
        AcquisitionConfig(
            name="nparego",
            acqf_cls=_NParEGOExpectedImprovement,
        ),
        DataContext(),
    )

    scalarized = resolved.objective(bundle.train_Y)
    assert scalarized.shape == torch.Size([bundle.train_Y.shape[0]])
    torch.testing.assert_close(context.best_f, scalarized.max().detach())


def test_explicit_nparego_objective_is_preserved() -> None:
    bundle = _make_bundle()
    explicit_objective = lambda samples, X=None: samples[..., 0]

    resolved, context = engine_defaults.resolve_acquisition_defaults(
        bundle,
        AcquisitionConfig(
            name="nparego",
            acqf_cls=_NParEGOExpectedImprovement,
            objective=explicit_objective,
        ),
        DataContext(),
    )

    assert resolved.objective is explicit_objective
    torch.testing.assert_close(context.best_f, bundle.train_Y[:, 0].max())


def test_explicit_nparego_best_f_is_preserved(monkeypatch) -> None:
    bundle = _make_bundle()
    explicit = torch.tensor(9.0, dtype=torch.double)
    monkeypatch.setattr(
        engine_defaults,
        "compute_best_f",
        lambda *args, **kwargs: pytest.fail("automatic best_f must not run"),
    )

    resolved, context = engine_defaults.resolve_acquisition_defaults(
        bundle,
        AcquisitionConfig(
            name="nparego",
            acqf_cls=_NParEGOExpectedImprovement,
            acqf_kwargs={"best_f": explicit},
        ),
        DataContext(best_f=torch.tensor(5.0, dtype=torch.double)),
    )

    assert resolved.objective is not None
    assert resolved.acqf_kwargs["best_f"] is explicit
    assert context.best_f is None
