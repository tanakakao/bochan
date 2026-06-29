from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from bochan.api import AcquisitionConfig, DataContext, ModelBundle, ModelConfig
import bochan.api.engine_defaults as engine_defaults


class _NParEGOExpectedImprovement:
    def __init__(self, model, best_f) -> None:
        self.model = model
        self.best_f = best_f


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
        model=SimpleNamespace(),
        train_X=train_X,
        train_Y=train_Y,
        model_config=config,
        task_type="regression",
        model_type="base",
    )


def test_nparego_computes_best_f_when_required(monkeypatch) -> None:
    bundle = _make_bundle()
    expected = torch.tensor(1.75, dtype=torch.double)
    calls = []

    def fake_compute_best_f(bundle_arg, config_arg, context_arg):
        calls.append((bundle_arg, config_arg, context_arg))
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
    assert context.best_f is expected
    assert "best_f" not in resolved.acqf_kwargs


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

    assert resolved.acqf_kwargs["best_f"] is explicit
    assert context.best_f is None
